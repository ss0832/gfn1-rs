"""Gamma-centred DFPT sampling, forces and ASE reproduce the same Hamiltonian."""
import numpy as np
from ase import Atoms
from ase.units import Hartree,Bohr
from gfn1_rs import Gfn1NativeCalculator,GFN1RSCalculator,FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM

def test_gamma_centered_crystal_hessian_and_force_derivative():
    a=3.567
    cell=np.array([[0,a/2,a/2],[a/2,0,a/2],[a/2,a/2,0]])
    positions=np.array([[0,0,0],[a/4,a/4,a/4]])
    settings=dict(electronic_temperature=300.,mixing=.1,max_scc=1000,energy_tolerance=1e-13,charge_tolerance=1e-11)
    calc=Gfn1NativeCalculator(**settings)
    h=np.asarray(calc.hessian_periodic([6,6],positions.tolist(),cell.tolist(),kgrid=(2,2,2),gamma_centered=True))
    eigen=np.linalg.eigvalsh(h)
    assert np.ptp(eigen[-3:])<1e-10
    legacy=np.asarray(calc.hessian_periodic([6,6],positions.tolist(),cell.tolist(),kgrid=(2,2,2)))
    assert np.max(abs(h-legacy))>1e-3
    atoms=Atoms("C2",positions=positions,cell=cell,pbc=True)
    atoms.calc=GFN1RSCalculator(**settings,kgrid=(2,2,2))
    np.testing.assert_allclose(atoms.calc.get_hessian(atoms=atoms,gamma_centered=True),h*Hartree/Bohr**2,atol=1e-9)
    from gfn1_rs import ANGSTROM_TO_BOHR
    r=positions*ANGSTROM_TO_BOHR;c=cell*ANGSTROM_TO_BOHR
    def gradient(step):
        moved=r.copy();moved[0,0]+=step
        result=calc.calculate_periodic([6,6],moved.tolist(),c.tolist(),unit="bohr",kgrid=(2,2,2),gamma_centered=True,compute_gradient=True)
        return -np.asarray(result.forces_ev_per_angstrom).ravel()/FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM
    step=1e-3
    coarse=(gradient(step)-gradient(-step))/(2*step)
    fine=(gradient(step/2)-gradient(-step/2))/step
    np.testing.assert_allclose(h[:,0],(4*fine-coarse)/3,atol=1e-8)
