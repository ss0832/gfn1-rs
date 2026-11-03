#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Explicit AO-cutoff comparison at one unchanged CIF-descendant geometry."""
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
    args=parser.parse_args()
    raw=args.reference_report.read_bytes()
    source=json.loads(raw)
    reference=source['operations']['joint_atomic_cell_relaxation']['result']
    assert reference['converged']
    geometry=dict(numbers=source['source_input']['numbers'],positions=reference['positions_bohr'],
        cell=reference['cell_bohr'],unit='bohr')
    grid=[(i/4,j/4,k/4) for i in range(4) for j in range(4) for k in range(4)]
    rows=[]
    for cutoff in [24.,30.,40.,50.]:
        calc=Gfn1NativeCalculator(param_path='builtin',electronic_temperature=300.,
            energy_tolerance=1e-13,charge_tolerance=1e-11,max_scc=1000,
            pbc_ao_cutoff=cutoff,threads=2,memory_mb=2048)
        start=time.perf_counter()
        try:
            bands=calc.band_structure(**geometry,kmesh=(2,2,2),gamma_centered=True,kpoints=grid)
            values=np.asarray(bands['energies_ev'])
            dos=calc.electronic_dos(bands['energies_hartree'],[1/64]*64,
                np.linspace(float(values.min())-1.,float(values.max())+1.,1001).tolist(),sigma_ev=.1)
            row=dict(ao_cutoff_bohr=cutoff,status='completed',bands=bands,electronic_dos=dos)
        except Exception as error:
            row=dict(ao_cutoff_bohr=cutoff,status='failed',error=f'{type(error).__name__}: {error}')
        row['elapsed_seconds']=time.perf_counter()-start
        rows.append(row)
        result=dict(schema='gfn1-rs-real-crystal-cutoff-comparison-v1',id=source['id'],
            reference_report_sha256=hashlib.sha256(raw).hexdigest(),geometry=geometry,
            fixed_conditions=dict(electronic_temperature_kelvin=300.,charge=0.,scc_mesh=[2,2,2],
                gamma_centered=True,postprocess_grid=[4,4,4],energy_tolerance_hartree=1e-13,
                charge_tolerance=1e-11,max_scc=1000,threads=2,workspace_memory_mb=2048),
            parameter_source=calc.param_source(),
            measured_extension_sha256=hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
            measured_program_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            attempts=rows)
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(cutoff,row['status'],row.get('error',''),flush=True)


if __name__=='__main__':
    main()
