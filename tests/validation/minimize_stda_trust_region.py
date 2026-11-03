#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unconstrained exact-Hessian trust-region searches of the original signed PES."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from ase.units import Bohr, Hartree
from gfn1_rs import Gfn1NativeCalculator
import gfn1_rs._native as extension
from search_stda_stationary import internal_space


def trust_step(eigenvalues, eigenvectors, gradient, radius):
    """Global quadratic trust-region minimizer, including the hard case."""
    g = eigenvectors.T @ gradient
    w = eigenvalues
    if w[0] > 0:
        newton = -g/w
        if np.linalg.norm(newton) <= radius:
            return eigenvectors @ newton
    lower = max(0., -w[0])
    tolerance = 1e-12*max(1., np.max(np.abs(w)))
    null = np.abs(w+lower) < tolerance
    regular = np.divide(-g, w+lower, out=np.zeros_like(g), where=~null)
    if np.linalg.norm(g[null]) < 1e-12 and np.linalg.norm(regular) <= radius:
        # Both signs have the same quadratic value in the true hard case.
        regular[0] = -np.copysign(np.sqrt(max(0.,radius**2-regular@regular)), g[0])
        return eigenvectors @ regular
    lo = lower
    hi = max(lower+1., 2*np.linalg.norm(g)/radius)
    while np.linalg.norm(g/(w+hi)) > radius:
        hi = 2*hi+1.
    for _ in range(90):
        shift = .5*(lo+hi)
        if np.linalg.norm(g/(w+shift)) > radius:
            lo = shift
        else:
            hi = shift
    return eigenvectors @ (-g/(w+hi))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('target/validation/stda-trust-region'))
    parser.add_argument('--steps',type=int,default=120)
    parser.add_argument('--cases',default='5:0:1.34,45:.15:1.45,75:.25:1.55,95:.25:1.45,115:-.2:1.5,150:.1:1.4')
    parser.add_argument('--fmax',type=float,default=1e-5)
    args=parser.parse_args()
    source=Path('tests/data/tda-stda-native-comparison.json')
    initial=next(c for c in json.loads(source.read_text())['cases'] if c['name']=='ethylene')
    numbers=initial['numbers']
    calc=Gfn1NativeCalculator(electronic_temperature=300.,energy_tolerance=1e-13,
        charge_tolerance=1e-11,max_scc=1000,threads=2,memory_mb=1024)
    args.output.mkdir(parents=True,exist_ok=True)
    for case_index,spec in enumerate(args.cases.split(',')):
        twist,pyramid,cc=map(float,spec.split(':'))
        r=np.asarray(initial['positions_angstrom']).copy()
        axis=r[1]-r[0];axis/=np.linalg.norm(axis)
        centre=r[:2].mean(axis=0)
        delta=(cc-np.linalg.norm(r[1]-r[0]))/2
        for i in range(len(r)):
            parent=0 if i==0 or (i>1 and np.linalg.norm(r[i]-r[0])<np.linalg.norm(r[i]-r[1])) else 1
            r[i]+=(2*parent-1)*delta*axis
        theta=np.deg2rad(twist)
        for i in (4,5):
            v=r[i]-r[1]
            r[i]=r[1]+v*np.cos(theta)+np.cross(axis,v)*np.sin(theta)+axis*(axis@v)*(1-np.cos(theta))
        r[0,2]+=pyramid
        r+=np.random.default_rng(717+case_index).normal(scale=.002,size=r.shape)
        r-=r.mean(axis=0)
        r/=Bohr
        path=args.output/f'case-{case_index}.json'
        start=time.monotonic()
        report=dict(schema='gfn1-rs-original-stda-trust-region-v1',package_version='0.7.0',
            model='unchanged signed F_GFN1+Omega_sTDA state0, 20 eV window',
            electronic_temperature_kelvin=300.,nuclear_constraints='none',
            algorithm='exact analytic Hessian trust-region quadratic minimization; negative-curvature escape',
            numbers=numbers,seed_parameters=dict(twist_degrees=twist,pyramid_angstrom=pyramid,cc_angstrom=cc),
            initial_positions_angstrom=(r*Bohr).tolist(),frames=[],status='running',
            measured_extension_sha256=hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
            input_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        def save():
            report['elapsed_seconds']=time.monotonic()-start
            path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        radius=.08/Bohr
        state=None
        rejected=0
        for iteration in range(args.steps):
            if state is None:
                state=calc.stda_xtb_gradient(numbers,(r*Bohr).tolist(),state=0,energy_window_ev=20.)
            gradient=np.asarray(state['gradient']).ravel()
            basis=internal_space(r)
            full=np.asarray(calc.stda_xtb_hessian(numbers,(r*Bohr).tolist(),state=0,energy_window_ev=20.)['hessian'])
            hessian=basis.T@((full+full.T)*.5)@basis
            g=basis.T@gradient
            w,v=np.linalg.eigh(hessian)
            fmax=float(np.linalg.norm(gradient.reshape(-1,3),axis=1).max()*Hartree/Bohr)
            frame=dict(iteration=iteration,positions_angstrom=(r*Bohr).tolist(),
                energy_ev=state['total_energy_hartree']*Hartree,excitation_ev=state['excitation_energy_hartree']*Hartree,
                fmax_ev_per_angstrom=fmax,minimum_internal_curvature_hartree_per_bohr2=float(w[0]),
                negative_modes=int(np.count_nonzero(w<-1e-6)),radius_bohr=radius,rejected_trials=rejected)
            report['frames'].append(frame);save()
            print(case_index,iteration,frame['energy_ev'],fmax,w[0],radius,flush=True)
            if fmax<args.fmax and w[0]>1e-6:
                report['status']='force_zero_positive_curvature_minimum';save();break
            accepted=False
            for trial_index in range(24):
                internal=trust_step(w,v,g,radius)
                step=basis@internal
                atom_max=np.linalg.norm(step.reshape(-1,3),axis=1).max()
                step*=min(1.,.04/Bohr/max(atom_max,1e-30))
                internal=basis.T@step
                predicted=-float(g@internal+.5*internal@hessian@internal)
                if predicted<=0:
                    report['status']='non_descent_quadratic_model';save();break
                trial=r+step.reshape(-1,3)
                try:
                    next_state=calc.stda_xtb_gradient(numbers,(trial*Bohr).tolist(),state=0,energy_window_ev=20.)
                    actual=state['total_energy_hartree']-next_state['total_energy_hartree']
                    ratio=actual/predicted
                except (ValueError,RuntimeError) as error:
                    ratio=-1.
                    frame['last_error']=str(error)
                if ratio<.25:
                    radius*=.25
                elif ratio>.75 and np.linalg.norm(internal)>.9*radius:
                    radius=min(2*radius,.25/Bohr)
                if ratio>.1 and actual>0:
                    r=trial;state=next_state;accepted=True
                    frame['accepted_ratio']=ratio
                    break
                rejected+=1
                if radius<1e-8/Bohr:
                    report['status']='trust_radius_floor_without_force_convergence';break
            save()
            if not accepted:
                if report['status']=='running':report['status']='trial_limit_without_force_convergence'
                save();break
        else:
            report['status']='iteration_limit_without_force_convergence';save()
        print(case_index,report['status'],flush=True)


if __name__=='__main__':
    main()
