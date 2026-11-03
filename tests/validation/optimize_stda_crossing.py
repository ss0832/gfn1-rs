#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Optimize an uncoupled GFN1/sTDA crossing with analytic gradients/Hessians.

Every nuclear coordinate is free. This solves crossing stationarity of the
upper envelope max(F_GFN1,F_GFN1+Omega), not force-zero stationarity of the
individual signed CI branch. No derivative coupling or physical CI is claimed.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from ase.units import Bohr,Hartree
from gfn1_rs import Gfn1NativeCalculator, __version__


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed",type=Path,help="Bundled crossing report or numbers/positions_angstrom JSON")
    parser.add_argument("--output",type=Path,default=Path("target/validation/stda-crossing"))
    parser.add_argument("--max-steps",type=int,default=100)
    args=parser.parse_args()
    if args.max_steps<=0:parser.error("max-steps must be positive")
    data=json.loads(args.seed.read_text(encoding="utf-8-sig"))
    previous=data.get("optimization",data)
    numbers=previous["numbers"]
    initial=previous.get("initial_positions_angstrom",previous.get("positions_angstrom"))
    if initial is None:raise ValueError("The seed must include explicit coordinates")
    r=np.asarray(initial,dtype=float)/Bohr
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    calc=Gfn1NativeCalculator(electronic_temperature=300.,max_scc=1000,energy_tolerance=1e-13,charge_tolerance=1e-11)
    report={'schema':'gfn1-rs-uncoupled-crossing-optimization-v1','electronic_temperature_kelvin':300.,'model':'GFN1 free energy and GFN1+sTDA signed CI branch',
            'geometry_constraints':'none','stationarity_condition':'Omega=0; g0+lambda*dOmega=0; lambda in [0,1] for upper-envelope minimum',
            'energy_window_ev':20.,'numbers':numbers,'initial_positions_angstrom':(r*Bohr).tolist(),'status':'running','frames':[]}
    report.update(package_version=__version__, input_sha256=hashlib.sha256(args.seed.read_bytes()).hexdigest(), initial_seed_source=str(args.seed), max_steps=args.max_steps)
    start=time.monotonic()
    def evaluate(pos):
        geometry=(pos*Bohr).tolist()
        excited=calc.stda_xtb_gradient(numbers,geometry,state=0,energy_window_ev=20.)
        reference=calc.calculate(numbers,geometry,compute_gradient=True)
        return reference.energy_hartree,np.asarray(reference.gradient_hartree_per_bohr).ravel(),excited['excitation_energy_hartree'],np.asarray(excited['excitation_gradient']).ravel(),excited
    def save():
        report['elapsed_seconds']=time.monotonic()-start
        (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for iteration in range(args.max_steps):
        energy,g0,gap,ggap,excited=evaluate(r)
        centered=r-r.mean(axis=0);rigid=[]
        for unit in np.eye(3):
            rigid.extend([np.tile(unit,(len(r),1)).ravel(),np.cross(np.tile(unit,(len(r),1)),centered).ravel()])
        _,singular,vh=np.linalg.svd(np.asarray(rigid),full_matrices=True)
        rank=int(np.count_nonzero(singular>singular[0]*1e-10));internal=vh[rank:].T
        g=internal.T@g0;normal=internal.T@ggap;nn=normal@normal
        if nn<1e-24:raise ValueError("The crossing gap gradient vanishes; the single-constraint stationarity system is singular")
        multiplier=float(-g@normal/nn)
        projected=internal@(g+multiplier*normal)
        fmax=float(np.linalg.norm(projected.reshape(-1,3),axis=1).max()*Hartree/Bohr)
        h=calc.stda_xtb_hessian(numbers,(r*Bohr).tolist(),state=0,energy_window_ev=20.)
        h0=np.asarray(h['ground_hessian']);hx=np.asarray(h['excitation_hessian']);hl=h0+multiplier*hx
        _,_,vh_gap=np.linalg.svd(normal.reshape(1,-1),full_matrices=True);tangent=vh_gap[1:].T
        ht=tangent.T@internal.T@hl@internal@tangent
        values,vectors=np.linalg.eigh((ht+ht.T)/2)
        frame={'iteration':iteration,'reference_energy_hartree':energy,'upper_energy_hartree':energy+max(0.,gap),
               'gap_ev':gap*Hartree,'lambda':multiplier,'projected_fmax_ev_per_angstrom':fmax,
               'branch_fmax_ev_per_angstrom':float(np.linalg.norm((g0+ggap).reshape(-1,3),axis=1).max()*Hartree/Bohr),
               'positions_angstrom':(r*Bohr).tolist(),'ground_gradient_hartree_per_bohr':g0.reshape(-1,3).tolist(),
               'gap_gradient_hartree_per_bohr':ggap.reshape(-1,3).tolist(),
               'seam_tangent_curvatures_hartree_per_bohr2':values.tolist()}
        report['frames'].append(frame);save()
        print(iteration,frame['upper_energy_hartree']*Hartree,frame['gap_ev'],multiplier,fmax,float(values[0]),flush=True)
        if abs(gap*Hartree)<1e-6 and fmax<1e-3 and values[0]>-1e-6:
            report['status']='crossing_stationary';report['upper_envelope_minimum_criterion']=bool(0.<multiplier<1. and values[0]>0.);save();break
        hn=internal.T@hl@internal
        dn=-gap*normal/nn
        gt=tangent.T@(g+hn@dn)
        dt=-vectors@((vectors.T@gt)/np.maximum(np.abs(values),1e-3))
        step=internal@(dn+tangent@dt)
        if np.linalg.norm(dt)<1e-6 and values[0]<-1e-6:
            step=internal@dn+internal@tangent@vectors[:,0]*(.01/Bohr)
        maximum=np.linalg.norm(step.reshape(-1,3),axis=1).max();step*=min(1.,.03/Bohr/max(maximum,1e-30))
        rho=max(1.,abs(multiplier)+1.);merit=energy+rho*abs(gap)
        slope=float(g0@step+rho*(abs(gap+ggap@step)-abs(gap)))
        accepted=False
        for backtrack in range(40):
            alpha=2.**(-backtrack);trial=r+alpha*step.reshape(-1,3)
            try:
                trial_energy,_,trial_gap,_,_=evaluate(trial)
                if trial_energy+rho*abs(trial_gap)<=merit+1e-4*alpha*min(slope,0.):
                    r=trial;accepted=True;break
            except (ValueError,RuntimeError) as error:report['last_rejected_trial']=str(error)
        frame['backtracks']=backtrack;save()
        if not accepted or np.linalg.norm(alpha*step)<1e-12:
            report['status']='line_search_stalled';save();break
    else:report['status']='step_limit';save()
    print(report['status'],report.get('upper_envelope_minimum_criterion'),flush=True)

if __name__=="__main__":main()
