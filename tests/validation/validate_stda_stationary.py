#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent full Cartesian checks of original signed-sheet force-zero points."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from ase.units import Bohr, Hartree
import gfn1_rs._native as extension
from ethylene_excited_optimization import ExcitedCalculator
from search_stda_stationary import internal_space
from ethylene_geometry import geometry


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=Path)
    args=parser.parse_args()
    source=json.loads(args.input.read_text())
    if not source['status'].startswith('force_zero') or source['geometry_constraints']!='none':
        raise ValueError('An unconstrained original-sheet force-zero result is required')
    r=np.asarray(source['frames'][-1]['positions_angstrom'])/Bohr
    calc=ExcitedCalculator('stda_xtb',20.)
    nums=source['numbers']
    result,spectrum=calc.evaluate(nums,(r*Bohr).tolist())
    configurations=spectrum['configurations']
    g=np.asarray(result['gradient']).ravel()
    analytic=np.asarray(calc.native.stda_xtb_hessian(nums,(r*Bohr).tolist(),state=0,energy_window_ev=20.)['hessian'])
    fdg=[];fdh=[]
    for step in (1e-4,5e-5):
        grad=np.zeros(r.size);hess=np.zeros((r.size,r.size))
        for y in range(r.size):
            values=[]
            for sign in (1.,-1.):
                trial=r.copy();trial.ravel()[y]+=sign*step
                value,lines=calc.evaluate(nums,(trial*Bohr).tolist())
                if sorted(lines['configurations'])!=sorted(configurations):
                    raise ValueError(f'The derivative stencil crossed a configuration boundary at coordinate {y}, sign {sign}, step {step}')
                values.append(value)
            grad[y]=(values[0]['total_energy_hartree']-values[1]['total_energy_hartree'])/(2*step)
            hess[:,y]=(np.asarray(values[0]['gradient']).ravel()-np.asarray(values[1]['gradient']).ravel())/(2*step)
        fdg.append(grad);fdh.append(hess)
    rg=(4*fdg[1]-fdg[0])/3;rh=(4*fdh[1]-fdh[0])/3
    internal=internal_space(r)
    eig,vectors=np.linalg.eigh(internal.T@((analytic+analytic.T)/2)@internal)
    fmax=float(np.linalg.norm(g.reshape(-1,3),axis=1).max()*Hartree/Bohr)
    errors=dict(gradient_hartree_per_bohr=float(np.abs(g-rg).max()),hessian_hartree_per_bohr2=float(np.abs(analytic-rh).max()))
    shape=geometry(dict(numbers=nums,accepted_frames=[dict(positions_angstrom=(r*Bohr).tolist())]))
    report=dict(schema='gfn1-rs-original-stda-stationary-audit-v1',package_version=source['package_version'],
        model=source['model'],temperature_kelvin=300.,geometry_constraints='none',status='pass',
        positions_angstrom=(r*Bohr).tolist(),numbers=nums,raw_fmax_ev_per_angstrom=fmax,
        energy_ev=result['total_energy_hartree']*Hartree,excitation_ev=result['excitation_energy_hartree']*Hartree,
        gradient_hartree_per_bohr=g.reshape(-1,3).tolist(),analytic_hessian_hartree_per_bohr2=analytic.tolist(),
        richardson_gradient=rg.tolist(),richardson_hessian=rh.tolist(),steps_bohr=[1e-4,5e-5],errors=errors,
        internal_curvatures_hartree_per_bohr2=eig.tolist(),negative_modes=int(np.count_nonzero(eig<-1e-6)),
        geometry=shape,all_components=dict(gradient=r.size,hessian=r.size*r.size),
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        measured_extension_sha256=hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest())
    if fmax>1e-5 or errors['gradient_hartree_per_bohr']>5e-8 or errors['hessian_hartree_per_bohr2']>5e-7:
        report['status']='fail'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','raw_fmax_ev_per_angstrom','errors','negative_modes','geometry')},indent=2),flush=True)
    if report['status']!='pass':raise ValueError('Original-sheet stationary-point verification failed')
    if args.seeds:
        args.seeds.mkdir(parents=True,exist_ok=True)
        direction=(internal@vectors[:,0]).reshape(-1,3)
        for sign,label in ((1.,'positive'),(-1.,'negative')):
            displaced=r+sign*(.04/Bohr)*direction/np.linalg.norm(direction,axis=1).max()
            seed=dict(numbers=nums,positions_angstrom=(displaced*Bohr).tolist(),
                provenance=f'Original signed-sheet saddle negative-curvature mode, {label}, max atomic displacement 0.04 angstrom; no geometric constraints',
                reference_audit_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
            (args.seeds/f'{label}.json').write_text(json.dumps(seed,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
