"""Installed native/ASE analytic excited Hessians and stable state identity."""
from pathlib import Path
import json
import numpy as np
import pytest
from ase import Atoms
from ase.units import Hartree, Bohr
from gfn1_rs import Gfn1NativeCalculator, GFN1RSCalculator


@pytest.mark.parametrize("model", ["tda", "stda_xtb"])
def test_native_and_ase_excited_hessian_components_and_units(model):
    numbers = [8,1,1]
    positions = [[.02,-.01,.03],[.958,.02,-.01],[-.25,.925,.04]]
    controls = dict(electronic_temperature=300.,max_scc=1000,
                    energy_tolerance=1e-13,charge_tolerance=1e-11)
    calc = Gfn1NativeCalculator(**controls)
    method = getattr(calc,model+"_hessian")
    options = {"energy_window_ev":20.} if model == "stda_xtb" else {}
    result = method(numbers,positions,**options)
    assert result["method"] == "analytic" and result["implementation"] == "native-rust"
    assert result["electronic_temperature_kelvin"] == 300.
    hessian = np.asarray(result["hessian"])
    assert hessian.shape == (9,9)
    np.testing.assert_array_equal(hessian,hessian.T)
    np.testing.assert_allclose(hessian,np.asarray(result["ground_hessian"])+result["excitation_hessian"],atol=1e-13)
    assert result["total_energy_hartree"] == result["ground_energy_hartree"]+result["excitation_energy_hartree"]
    atoms=Atoms(numbers=numbers,positions=positions)
    atoms.calc=GFN1RSCalculator(**controls)
    ase=getattr(atoms.calc,"get_"+model+"_hessian")(atoms=atoms,**options)
    np.testing.assert_allclose(ase["hessian"],hessian*Hartree/Bohr**2,atol=1e-9)
    np.testing.assert_allclose(ase["hessian_hartree_per_bohr2"],hessian,atol=1e-12)
    assert abs(ase["total_energy"]-result["total_energy_hartree"]*Hartree)<1e-10
    with pytest.raises(ValueError,match="budget"):
        method(numbers,positions,memory_budget_bytes=1,**options)
    atoms.set_cell([12.,12.,12.]); atoms.pbc=True
    with pytest.raises(ValueError,match="require a molecule"):
        getattr(atoms.calc,"get_"+model+"_hessian")(atoms=atoms,**options)


def test_stda_instability_is_visible_and_signed_derivatives_keep_the_root():
    data=json.loads((Path(__file__).resolve().parents[1]/"data/stda-ci-instability.json").read_text())
    calc=Gfn1NativeCalculator(electronic_temperature=300.)
    spectrum=calc.stda_xtb(data["numbers"],data["positions_angstrom"],energy_window_ev=20.)
    assert spectrum["nonpositive_ci_roots"]>0
    assert spectrum["lowest_ci_eigenvalue_hartree"]<0
    assert spectrum["excitation_energies_hartree"][0]>0
    assert spectrum["ci_eigenvalues_hartree"][0]<0
    for method in (calc.stda_xtb_gradient,calc.stda_xtb_hessian):
        result=method(data["numbers"],data["positions_angstrom"],energy_window_ev=20.)
        assert result["nonpositive_ci_roots"]>0
        assert abs(result["excitation_energy_hartree"]-spectrum["ci_eigenvalues_hartree"][0])<1e-12
