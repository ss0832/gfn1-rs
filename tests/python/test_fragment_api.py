"""Molecular/periodic native and ASE fragment API contracts at identical 300 K."""
import numpy as np
import pytest
from ase import Atoms
from ase.units import Hartree,Bohr
from gfn1_rs import Gfn1NativeCalculator,GFN1RSCalculator

@pytest.mark.parametrize("periodic",[False,True])
def test_fragment_native_and_ase_units(periodic):
    nums=[1,9,1,9,1,9]
    pos=[[0,0,0],[.92,0,0],[0,3,0],[.92,3,0],[3,0,0],[3.92,0,0]]
    fragments=[[0,1],[2,3],[4,5]]
    settings=dict(electronic_temperature=300.,max_scc=1000,energy_tolerance=1e-13,charge_tolerance=1e-11)
    calc=Gfn1NativeCalculator(**settings)
    atoms=Atoms(numbers=nums,positions=pos,cell=[12,12,12] if periodic else None,pbc=periodic)
    atoms.calc=GFN1RSCalculator(**settings)
    geometry=dict(cell=atoms.cell.tolist()) if periodic else {}
    for method in ("fmo_xtb","divide_and_conquer"):
        options=dict(fragment_charges=[0,0,0])
        if method=="divide_and_conquer":options["buffers"]=[list(range(6))]*3
        native=getattr(calc,method)(nums,pos,fragments,**options,**geometry)
        ase=getattr(atoms.calc,"get_"+method)(fragments,atoms=atoms,**options)
        assert abs(ase["total_free"]-native["total_free_hartree"]*Hartree)<1e-10
        np.testing.assert_allclose(ase["forces"],np.asarray(native["forces"])*Hartree/Bohr,atol=1e-10)
        if periodic:
            np.testing.assert_allclose(ase["stress"],np.asarray(native["stress"])*Hartree/Bohr**3,atol=1e-10)
        else:assert ase["stress"] is None
        no_derivatives=getattr(atoms.calc,"get_"+method)(fragments,atoms=atoms,compute_derivatives=False,**options)
        assert no_derivatives["gradient"] is None and no_derivatives["forces"] is None and no_derivatives["stress"] is None
    if periodic:
        atoms.pbc=(True,True,False)
        with pytest.raises(ValueError,match="3D"):
            atoms.calc.get_fmo_xtb(fragments,atoms=atoms)
