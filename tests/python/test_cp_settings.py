"""User CP controls reach real direct solves, logs, ASE and both pip commands."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from ase import Atoms
from gfn1_rs import Gfn1NativeCalculator,GFN1RSCalculator

NUMS=[8,1,1]
POS=[[.02,-.01,.03],[.958,.02,-.01],[-.25,.925,.04]]

def test_native_ase_cp_settings_and_actual_residual(tmp_path):
    options=dict(cp_tolerance=1e-10,cp_relative_tolerance=1e-10,cp_max_iter=321)
    calc=Gfn1NativeCalculator(**options,log_file=str(tmp_path/'cp.jsonl'))
    assert calc.cp_settings()==dict(absolute_tolerance=1e-10,relative_tolerance=1e-10,max_iterations=321)
    hessian=calc.hessian(NUMS,POS)
    reference=Gfn1NativeCalculator().hessian(NUMS,POS)
    np.testing.assert_allclose(hessian,reference,atol=1e-11,rtol=0)
    records=[json.loads(line) for line in (tmp_path/'cp.jsonl').read_text().splitlines()]
    assert any(r.get('data',{}).get('configuration',{}).get('cp_settings',{}).get('max_iterations')==321 for r in records)
    atoms=Atoms(numbers=NUMS,positions=POS)
    atoms.calc=GFN1RSCalculator(**options)
    atoms.calc.get_hessian(atoms)
    assert atoms.calc._native.cp_settings()==calc.cp_settings()
    atoms.calc.set(cp_max_iter=456)
    assert atoms.calc._native.cp_settings()['max_iterations']==456
    strict=Gfn1NativeCalculator(cp_tolerance=1e-30,cp_relative_tolerance=1e-30)
    with pytest.raises(ValueError,match='CP residual'):
        strict.hessian(NUMS,POS)
    soc_options=dict(zeta_hartree=[(8,1,.02)], parameter_provenance='independent CP control regression',
                     compute_forces=False,compute_response=True)
    soc=calc.soc_calculate(NUMS,POS,**soc_options)
    assert soc['response_normalized_residual']<1e-13
    with pytest.raises(ValueError,match='spinor CP-SCC residual'):
        strict.soc_calculate(NUMS,POS,**soc_options)
    for arguments in ({'cp_tolerance':0.},{'cp_relative_tolerance':float('nan')},{'cp_max_iter':0}):
        with pytest.raises(ValueError):Gfn1NativeCalculator(**arguments)

@pytest.mark.parametrize('name',['gfn1-rs','gfn1-rs-train'])
def test_cp_cli_controls_are_shared_with_native_commands(name,tmp_path):
    executable=Path(sys.executable).parent/(name+('.exe' if sys.platform=='win32' else ''))
    if name=='gfn1-rs':
        xyz=tmp_path/'water.xyz';xyz.write_text('3\nwater\nO .02 -.01 .03\nH .958 .02 -.01\nH -.25 .925 .04\n')
        args=[str(xyz),'--hessian']
    else:
        args=['new','--species','H','--centers','1.2,2.5','--cutoff','6','--output',str(tmp_path/'initial.model')]
    completed=subprocess.run([str(executable),*args,'--cp-tolerance','1e-10','--cp-relative-tolerance','1e-10','--cp-max-iter','321'],capture_output=True,text=True,encoding='utf-8')
    assert completed.returncode==0,completed.stderr
    assert 'CP settings:' in completed.stderr and '321' in completed.stderr
