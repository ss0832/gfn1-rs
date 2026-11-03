#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Execute property families on attributed CIF-derived real-crystal structures.

This is a diagnostic, not a chemical-accuracy pass gate. Unstable, nonstationary,
unconverged or out-of-window calculations retain their actual errors/status.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import gfn1_rs
import gfn1_rs._native as extension
from gfn1_rs import Gfn1NativeCalculator, BOHR_TO_ANGSTROM, HARTREE_TO_EV


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,default=Path('tests/data/real-crystal-survey-inputs.json'))
    parser.add_argument('--case',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['all','ground','relax','response','spectra','phonons','qha'],default='all')
    parser.add_argument('--mesh',type=int,default=2)
    parser.add_argument('--kmesh',type=int,nargs=3)
    parser.add_argument('--repetitions',type=int,nargs=3,default=[2,1,1])
    parser.add_argument('--cutoff',type=float,default=24.)
    parser.add_argument('--max-iterations',type=int,default=200)
    args=parser.parse_args()
    launched_program_sha256=sha(__file__)
    source=next(v for v in json.loads(args.inputs.read_text())['cases'] if v['id']==args.case)
    kmesh=tuple(args.kmesh or [args.mesh]*3)
    repetitions=tuple(args.repetitions)
    context=dict(electronic_temperature_kelvin=300.,charge=0.,
        electronic_mesh=list(kmesh),gamma_centered=True,ao_cutoff_bohr=args.cutoff,
        energy_tolerance_hartree=1e-13,charge_tolerance=1e-11,max_scc=1000,
        threads=2,workspace_memory_mb=2048,ml_model_identity=None)
    report=json.loads(args.output.read_text()) if args.output.exists() else dict(
        schema='gfn1-rs-sourced-real-crystal-property-executions-v1',
        package_version=gfn1_rs.__version__,id=source['id'],source_input=source,
        source_input_file_sha256=sha(args.inputs),conditions=context,operations={},
        measured_extension_sha256=sha(extension.__file__),
        validation_program_sha256=launched_program_sha256)
    assert report['conditions']==context and report['source_input']==source
    args.output.parent.mkdir(parents=True,exist_ok=True)

    def save():
        args.output.write_text(json.dumps(plain(report),indent=2,allow_nan=False)+'\n',encoding='utf-8')

    def operation(name, callback, conditions=None):
        if name in report['operations'] and report['operations'][name]['status']!='running':
            requested=conditions or context
            if report['operations'][name].get('conditions')!=requested:
                raise ValueError(f'Cached operation {name} has different conditions; choose a new output file to preserve both executions')
            return report['operations'][name].get('result')
        record=dict(status='running',conditions=conditions or context,
            measured_extension_sha256=sha(extension.__file__),
            validation_program_sha256=launched_program_sha256)
        report['operations'][name]=record
        save()
        start=time.perf_counter()
        try:
            result=plain(callback())
            record.update(status='completed',result=result)
            if isinstance(result,dict) and result.get('converged') is False:
                record['status']='not_converged'
        except Exception as error:
            record.update(status='failed',error=f'{type(error).__name__}: {error}')
            result=None
        record['elapsed_seconds']=time.perf_counter()-start
        save()
        print(f"{source['id']} {name}: {record['status']} ({record['elapsed_seconds']:.3f}s)",flush=True)
        return result

    calc=Gfn1NativeCalculator(param_path='builtin',electronic_temperature=300.,energy_tolerance=1e-13,
        charge_tolerance=1e-11,max_scc=1000,scc_accelerator='auto',
        pbc_ao_cutoff=args.cutoff,threads=2,memory_mb=2048)
    report['calculator_identity']=dict(parameter_source=calc.param_source(),ml_model_identity=calc.ml_model_info())
    supplied=dict(numbers=source['numbers'],positions=source['positions_angstrom'],
        cell=source['cell_angstrom'],kmesh=kmesh,gamma_centered=True)
    masses=source['masses_amu']

    def ground():
        arguments={k:v for k,v in supplied.items() if k!='kmesh'}
        arguments['kgrid']=supplied['kmesh']
        value=calc.calculate_periodic(**arguments,compute_gradient=True,compute_stress=True)
        return dict(converged=value.converged,free_energy_hartree=value.energy_hartree,
            energy_terms_hartree=value.energy_terms_hartree(),
            gradient_hartree_per_bohr=value.gradient_hartree_per_bohr,
            stress_hartree_per_bohr3=value.stress,charges=value.charges,iterations=value.iterations)

    if args.stage in ('all','ground'):
        operation('supplied_geometry_energy_forces_stress',ground)
    if args.stage in ('all','relax'):
        operation('joint_atomic_cell_relaxation',lambda:calc.relax_periodic(**supplied,
            relax_cell=True,max_iterations=args.max_iterations,
            force_tolerance=1e-7,stress_tolerance=1e-7))
    relaxed=report['operations'].get('joint_atomic_cell_relaxation',{}).get('result')
    geometry=supplied.copy()
    if relaxed and relaxed['converged']:
        geometry.update(positions=relaxed['positions_bohr'],cell=relaxed['cell_bohr'],unit='bohr')
    report['response_reference']='model_relaxed_CIF_descendant' if relaxed and relaxed['converged'] else 'supplied_CIF_geometry'
    volume=abs(np.linalg.det(geometry['cell']))
    if geometry.get('unit')!='bohr':
        volume/=BOHR_TO_ANGSTROM**3
    density=sum(masses)*1.66053906660e-27/(volume*(BOHR_TO_ANGSTROM*1e-10)**3)

    if args.stage in ('all','response'):
        elastic=operation('analytic_elasticity_internal_strain',lambda:calc.elastic_constants(**geometry))
        if elastic and elastic['relaxed_stiffness_hartree_per_bohr3'] is not None:
            c=elastic['relaxed_stiffness_hartree_per_bohr3']
            operation('compliance_vrh_moduli_sound',lambda:calc.elastic_properties(c,density))
            operation('directional_elasticity_sound',lambda:calc.directional_elastic_properties(c,density,(1,0,0),(0,1,0)))
        else:
            reason=(elastic or {}).get('relaxed_response_error','elastic reference unavailable')
            for name in ('compliance_vrh_moduli_sound','directional_elasticity_sound'):
                report['operations'][name]=dict(status='unavailable',reason=reason)
                save()
        operation('equation_of_state',lambda:calc.equation_of_state(**geometry,
            volume_ratios=[.94,.97,1.,1.03,1.06],relax_ions=True,max_iterations=args.max_iterations))
        electric=operation('born_proper_piezo_dielectric_ir',lambda:calc.electromechanical_response(**geometry,masses_amu=masses))
        if electric:
            for direction,label in [((1,0,0),'100'),((1,1,0),'110'),((1,1,1),'111')]:
                operation('lo_to_'+label,lambda direction=direction:calc.lo_to_modes(
                    electric['elastic']['nuclear_hessian_hartree_per_bohr2'],
                    electric['berry']['born_effective_charges'],
                    electric['electronic']['electronic_relative_dielectric'],volume,masses,direction))
    if args.stage in ('all','spectra'):
        knots=np.array([[0,0,0],[.5,0,.5],[.5,.5,.5],[0,0,0]],float)
        path=np.concatenate([np.linspace(a,b,13,endpoint=False) for a,b in zip(knots[:-1],knots[1:])]+[knots[-1:]])
        operation('electronic_band_path',lambda:calc.band_structure(**geometry,kpoints=[tuple(k) for k in path]))
        grid=[(i/4,j/4,k/4) for i in range(4) for j in range(4) for k in range(4)]
        sampled=operation('electronic_uniform_4cubed_sampling',lambda:calc.band_structure(**geometry,kpoints=grid),
            context|dict(fixed_density_evaluation_mesh=[4,4,4]))
        if sampled:
            energies=np.asarray(sampled['energies_ev'])
            lo,hi=float(energies.min())-1.,float(energies.max())+1.
            operation('electronic_density_of_states',lambda:calc.electronic_dos(
                sampled['energies_hartree'],[1/64]*64,np.linspace(lo,hi,1001).tolist(),sigma_ev=.1),
                context|dict(fixed_density_evaluation_mesh=[4,4,4],gaussian_broadening_ev=.1))
    if args.stage in ('all','phonons','qha'):
        phonon_args={k:v for k,v in geometry.items() if k not in ('kmesh','gamma_centered')}
        phonon_condition=context|dict(electronic_sampling='Gamma of supercell; primitive electronic folding follows repetitions',
            electronic_mesh=list(repetitions),reference_primitive_relaxation_mesh=list(kmesh),
            supercell_repetitions=list(repetitions),response_reference=report['response_reference'])
        mesh=operation('commensurate_phonon_mesh',lambda:calc.phonon_mesh(**phonon_args,
            repetitions=repetitions,masses_amu=masses,memory_budget_bytes=2048*1024*1024,max_atoms=32),phonon_condition)
        if mesh:
            frequencies=np.asarray(mesh['wavenumbers_cm1'])
            operation('phonon_density_of_states',lambda:calc.phonon_dos(mesh['wavenumbers_cm1'],mesh['weights'],
                np.linspace(float(frequencies.min())-50,float(frequencies.max())+50,801).tolist(),sigma_cm1=10.),phonon_condition)
            operation('harmonic_thermodynamics',lambda:calc.harmonic_thermodynamics(mesh['wavenumbers_cm1'],mesh['weights'],[0.,100.,300.,600.]),phonon_condition)
        if args.stage in ('all','qha'):
            samples=[]
            for ratio in [.88,.94,1.,1.06,1.12]:
                scale=ratio**(1/3)
                scaled=phonon_args|dict(positions=(np.asarray(phonon_args['positions'])*scale).tolist(),
                    cell=(np.asarray(phonon_args['cell'])*scale).tolist())
                value=operation(f'qha_volume_{ratio:g}',lambda scaled=scaled:calc.phonon_mesh(**scaled,
                    repetitions=repetitions,masses_amu=masses,memory_budget_bytes=2048*1024*1024,max_atoms=32),phonon_condition|dict(volume_ratio=ratio))
                if value:
                    samples.append((volume*ratio,value))
            if len(samples)==5:
                operation('quasiharmonic_zero_pressure',lambda:calc.quasiharmonic_zero_pressure(
                    [v for v,_ in samples],[v['reference_energy_hartree_per_primitive_cell'] for _,v in samples],
                    [v['wavenumbers_cm1'] for _,v in samples],samples[0][1]['weights'],[0.,100.,300.,600.]),
                    phonon_condition|dict(static_energy_sampling='same supercell SCC state as each phonon mesh',volume_ratios=[.88,.94,1.,1.06,1.12]))
    save()


if __name__=='__main__':
    main()
