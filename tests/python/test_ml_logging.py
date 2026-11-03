# SPDX-License-Identifier: GPL-3.0-or-later
"""Installed-wheel contracts for embedded ML models and operational logging."""
import json

import gfn1_rs
import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree
from gfn1_rs.ase import GFN1RSCalculator


NUMBERS = [8, 1, 1]
POSITIONS = [[0., 0., 0.], [0.958, 0., 0.], [-0.25, 0.925, 0.03]]


def settings():
    return dict(ml_model_path="builtin:water-clusters", electronic_temperature=0.,
                energy_tolerance=1e-12, charge_tolerance=1e-10)


def test_installed_model_ase_units_and_jsonl_calculation_identity(tmp_path):
    log = tmp_path / "water.jsonl"
    native = gfn1_rs.Gfn1NativeCalculator(**settings(), log_file=str(log), log_level="debug")
    result = native.calculate(NUMBERS, POSITIONS, compute_gradient=True)
    assert result.ml_model_sha256 == "5450b55436e10608a22064cfeb03541865d7efd8d9ab4f56d450b36514bbf819"
    assert "Adamo1999PBE0" in result.method_citations
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert any(r["operation_id"] == result.calculation_id and r["event"] == "operation_end"
               and r["data"]["status"] == "success" for r in records)
    assert any(r["event"] == "scf_iteration" for r in records)
    assert str(tmp_path) not in log.read_text()
    atoms = Atoms(numbers=NUMBERS, positions=POSITIONS)
    atoms.calc = GFN1RSCalculator(**settings())
    assert np.isclose(atoms.get_potential_energy(), result.energy_hartree * Hartree, atol=1e-11, rtol=0.)
    assert np.allclose(atoms.get_forces(), -np.array(result.gradient_hartree_per_bohr) * Hartree / Bohr,
                       atol=1e-10, rtol=0.)
    assert "Adamo1999PBE0" in atoms.calc.results["method_citations"]


def test_model_cubic_contractions_are_hessian_directional_derivatives():
    native = gfn1_rs.Gfn1NativeCalculator(**settings())
    positions = np.array(POSITIONS) / gfn1_rs.BOHR_TO_ANGSTROM
    direction = np.array([.2, -.1, .05, -.3, .2, -.1, .1, -.1, .05])
    cubic = native.third_derivative_contractions(NUMBERS, positions.tolist(),
                                                  [direction.tolist(), (-direction).tolist()], unit="bohr")
    h = 2e-4
    displacement = h * direction.reshape(-1, 3)
    plus = np.array(native.hessian(NUMBERS, (positions + displacement).tolist(), unit="bohr"))
    minus = np.array(native.hessian(NUMBERS, (positions - displacement).tolist(), unit="bohr"))
    reference = direction @ ((plus - minus) / (2 * h)) @ direction
    assert abs(cubic[0] - reference) < 3e-6
    assert abs(cubic[0] + cubic[1]) < 1e-11


def test_periodic_model_metadata_forces_and_stress_reach_the_ase_layer():
    cell = [[6., 0., 0.], [.2, 6.3, 0.], [0., .1, 6.6]]
    native = gfn1_rs.Gfn1NativeCalculator(**settings())
    result = native.calculate_periodic(NUMBERS, POSITIONS, cell, (True, True, True),
                                        kgrid=(2, 1, 1), compute_gradient=True, compute_stress=True)
    assert result.ml_model_sha256 == "5450b55436e10608a22064cfeb03541865d7efd8d9ab4f56d450b36514bbf819"
    assert "Buccheri2025Periodic" in result.method_citations
    assert "Adamo1999PBE0" in result.method_citations
    atoms = Atoms(numbers=NUMBERS, positions=POSITIONS, cell=cell, pbc=True)
    atoms.calc = GFN1RSCalculator(**settings(), kgrid=(2, 1, 1))
    forces_au = np.array(result.forces_ev_per_angstrom) / gfn1_rs.FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM
    assert np.allclose(atoms.get_forces(), forces_au * Hartree / Bohr, atol=2e-9, rtol=0.)
    assert np.allclose(atoms.get_stress(voigt=False), np.array(result.stress) * Hartree / Bohr**3,
                       atol=2e-10, rtol=0.)
    assert atoms.calc.results["ml_model_sha256"] == result.ml_model_sha256


def test_periodic_model_hessian_and_commensurate_phonon_bindings():
    native = gfn1_rs.Gfn1NativeCalculator(**settings())
    positions = np.array(POSITIONS) / gfn1_rs.BOHR_TO_ANGSTROM
    cell = (np.diag([4., 4.2, 4.4]) / gfn1_rs.BOHR_TO_ANGSTROM).tolist()
    hessian = np.array(native.hessian_periodic(NUMBERS, positions.tolist(), cell, kgrid=(2, 1, 1), unit="bohr"))
    assert hessian.shape == (9, 9)
    step = 1e-4
    plus, minus = positions.copy(), positions.copy()
    plus[1, 1] += step
    minus[1, 1] -= step
    def gradient(x):
        result = native.calculate_periodic(NUMBERS, x.tolist(), cell, (True, True, True),
                                           kgrid=(2, 1, 1), unit="bohr", compute_gradient=True)
        return -np.array(result.forces_ev_per_angstrom).reshape(-1) / gfn1_rs.FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM
    reference = (gradient(plus) - gradient(minus)) / (2 * step)
    assert np.max(np.abs(reference - hessian[:, 4])) < 3e-6
    phonons = native.phonons_commensurate(NUMBERS, positions.tolist(), cell, (2, 1, 1),
                                          [(.5, 0., 0.)], unit="bohr", masses_amu=[16., 1., 1.])
    assert phonons[0]["raw_hermiticity_error"] < 1e-11
    assert phonons[0]["raw_hessian_asymmetry_hartree_per_bohr2"] < 2e-7
    assert "Li2026MLxTB" in phonons[0]["method_citations"]
    assert len(phonons[0]["wavenumbers_cm1"]) == 9
