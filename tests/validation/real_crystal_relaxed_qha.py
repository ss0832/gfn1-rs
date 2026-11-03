#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Relax internal atoms at every fixed-volume node before consistent QHA sampling."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import gfn1_rs._native as extension
from gfn1_rs import Gfn1NativeCalculator


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference_report',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--force-tolerance',type=float,default=1e-7)
    args=parser.parse_args()
    raw=args.reference_report.read_bytes()
    reference=json.loads(raw)
    system=reference['operations']['joint_atomic_cell_relaxation']['result']
    assert system['converged']
    reps=reference['conditions']['electronic_mesh']
    mesh_label='x'.join(str(n) for n in reps)
    calc=Gfn1NativeCalculator(param_path='builtin',electronic_temperature=300.,
        energy_tolerance=1e-13,charge_tolerance=1e-11,max_scc=1000,
        pbc_ao_cutoff=reference['conditions']['ao_cutoff_bohr'],threads=2,memory_mb=2048)
    report=dict(schema='gfn1-rs-real-crystal-internal-relaxed-qha-v1',id=reference['id'],
        reference_report_sha256=hashlib.sha256(raw).hexdigest(),
        source_input=reference['source_input'],reference_conditions=reference['conditions'],
        parameter_source=calc.param_source(),electronic_temperature_kelvin=300.,
        internal_force_tolerance_hartree_per_bohr=args.force_tolerance,
        electronic_sampling=f'same primitive {mesh_label} Gamma-centred mesh in fixed-cell ionic relaxation and Gamma of{mesh_label} supercell for phonons; consistent static energies from supercell SCC',
        q_sampling=f'coarse {mesh_label} commensurate q mesh; no converged3D thermodynamics claim',
        volume_strain='isotropic fixed-shape volume scan; internal atoms independently relaxed at each node; no anisotropic thermal expansion',
        measured_extension_sha256=hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
        measured_program_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),samples=[])
    def save():
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    original_volume=abs(np.linalg.det(system['cell_bohr']))
    # Check the matching central3D reference before spending work on a volume scan.
    # The measured earlier runs retain their original launch-program hashes.
    center=dict(volume_ratio=1.,volume_bohr3=original_volume,status='running')
    report['reference_phonon_preflight']=center
    save()
    start=time.perf_counter()
    try:
        geometry=dict(numbers=reference['source_input']['numbers'],unit='bohr',
            positions=system['positions_bohr'],cell=system['cell_bohr'])
        relaxed=calc.relax_periodic(**geometry,kmesh=tuple(reps),gamma_centered=True,
            relax_cell=False,max_iterations=100,force_tolerance=args.force_tolerance)
        center['internal_relaxation']=relaxed
        if not relaxed['converged']:
            raise ValueError(f"Central internal relaxation did not converge: {relaxed['termination']}")
        geometry.update(positions=relaxed['positions_bohr'],cell=relaxed['cell_bohr'])
        phonons=calc.phonon_mesh(**geometry,repetitions=tuple(reps),
            masses_amu=reference['source_input']['masses_amu'],max_atoms=32,
            memory_budget_bytes=2048*1024*1024)
        center['phonons']=phonons
        if np.min(phonons['wavenumbers_cm1']) < -1e-3:
            raise ValueError('Central phonon reference has an imaginary mode; no stable QHA volume scan is attempted')
        center['status']='completed'
    except Exception as error:
        center.update(status='failed',error=f'{type(error).__name__}: {error}')
    center['elapsed_seconds']=time.perf_counter()-start
    save()
    if center['status']!='completed':
        return
    for ratio in [.88,.94,1.,1.06,1.12]:
        if ratio==1.:
            report['samples'].append(center|dict(reused_reference_preflight=True))
            save()
            continue
        start=time.perf_counter()
        sample=dict(volume_ratio=ratio,volume_bohr3=original_volume*ratio,status='running')
        report['samples'].append(sample)
        save()
        scale=ratio**(1/3)
        geometry=dict(numbers=reference['source_input']['numbers'],unit='bohr',
            positions=(np.asarray(system['positions_bohr'])*scale).tolist(),
            cell=(np.asarray(system['cell_bohr'])*scale).tolist())
        try:
            relaxed=calc.relax_periodic(**geometry,kmesh=tuple(reps),gamma_centered=True,
                relax_cell=False,max_iterations=100,force_tolerance=args.force_tolerance)
            sample['internal_relaxation']=relaxed
            if not relaxed['converged']:
                raise ValueError(f"Internal relaxation did not converge: {relaxed['termination']}")
            geometry.update(positions=relaxed['positions_bohr'],cell=relaxed['cell_bohr'])
            phonons=calc.phonon_mesh(**geometry,repetitions=tuple(reps),
                masses_amu=reference['source_input']['masses_amu'],max_atoms=32,
                memory_budget_bytes=2048*1024*1024)
            sample.update(status='completed',phonons=phonons)
        except Exception as error:
            sample.update(status='failed',error=f'{type(error).__name__}: {error}')
        sample['elapsed_seconds']=time.perf_counter()-start
        save()
        print(reference['id'],ratio,sample['status'],sample['elapsed_seconds'],flush=True)
    if all(s['status']=='completed' for s in report['samples']):
        try:
            data=report['samples']
            report['quasiharmonic_zero_pressure']=dict(status='completed',result=calc.quasiharmonic_zero_pressure(
                [s['volume_bohr3'] for s in data],
                [s['phonons']['reference_energy_hartree_per_primitive_cell'] for s in data],
                [s['phonons']['wavenumbers_cm1'] for s in data],data[0]['phonons']['weights'],[0.,100.,300.,600.]))
        except Exception as error:
            report['quasiharmonic_zero_pressure']=dict(status='failed',error=f'{type(error).__name__}: {error}')
    save()


if __name__=='__main__':
    main()
