# SPDX-License-Identifier: GPL-3.0-or-later
"""Physical spinor counting and native/ASE periodic SOC property contracts."""
import gfn1_rs
import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree
from gfn1_rs.ase import GFN1RSCalculator

NUMBERS = [8, 1, 1]
POSITIONS = [[0., 0., 0.], [.958, 0., 0.], [-.25, .925, .03]]
CELL = [[3.8, 0., 0.], [.2, 4., 0.], [0., .1, 4.2]]
ZETA = [(8, 1, .005)]
PROVENANCE = "Independent synthetic zeta L.S binding test; not a literature atomic constant"
SETTINGS = dict(electronic_temperature=0., energy_tolerance=1e-12, charge_tolerance=1e-10)


def test_periodic_soc_native_and_ase_units_share_a_physical_spinor_state():
    native = gfn1_rs.Gfn1NativeCalculator(**SETTINGS)
    result = native.soc_calculate_periodic(NUMBERS, POSITIONS, CELL, ZETA, PROVENANCE,
                                           kgrid=(2, 1, 1), compute_forces=True, compute_stress=True, compute_hessian=True)
    assert result["occupation_limit"] == 1
    assert sum(w * sum(o) for w, o in zip(result["kweights"], result["occupations_k"])) == 8.
    assert all(0. <= f <= 1. for occ in result["occupations_k"] for f in occ)
    assert "Jha2023SOC" in result["method_citations"]
    assert result["max_response_residual"] < 1e-9
    assert result["raw_hessian_asymmetry_hartree_per_bohr2"] < 2e-7
    atoms = Atoms(numbers=NUMBERS, positions=POSITIONS, cell=CELL, pbc=True)
    calc = GFN1RSCalculator(**SETTINGS, kgrid=(2, 1, 1))
    ase = calc.get_soc(ZETA, PROVENANCE, atoms, compute_forces=True, compute_stress=True, compute_hessian=True)
    assert abs(ase["energy"] - result["energy_hartree"] * Hartree) < 1e-10
    assert np.allclose(ase["forces"], np.array(result["forces_hartree_per_bohr"]) * Hartree / Bohr, atol=1e-9, rtol=0.)
    assert np.allclose(ase["stress"], np.array(result["stress_hartree_per_bohr3"]) * Hartree / Bohr**3, atol=1e-10, rtol=0.)
    assert np.allclose(ase["hessian"], np.array(result["hessian_hartree_per_bohr2"]) * Hartree / Bohr**2, atol=1e-9, rtol=0.)


def test_soc_commensurate_phonons_keep_spinor_response_diagnostics():
    native = gfn1_rs.Gfn1NativeCalculator(**SETTINGS)
    modes = native.phonons_commensurate(NUMBERS, POSITIONS, CELL, (2, 1, 1), [(.5, 0., 0.)],
                                        zeta_hartree=ZETA, parameter_provenance=PROVENANCE)
    assert modes[0]["max_spinor_response_residual"] < 1e-9
    assert modes[0]["raw_hessian_asymmetry_hartree_per_bohr2"] < 2e-7
    assert modes[0]["raw_hermiticity_error"] < 1e-11
    assert "Jha2023SOC" in modes[0]["method_citations"]
    assert len(modes[0]["wavenumbers_cm1"]) == 9
