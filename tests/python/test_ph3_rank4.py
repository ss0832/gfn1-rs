import json
from pathlib import Path

import numpy as np

from gfn1_rs import Gfn1NativeCalculator


def test_ph3_cold_rank4_gradient_matches_independent_cartesian_audit():
    reference = json.loads(
        (Path(__file__).resolve().parents[1] / "data" / "multipole-rank4-ph3-diagnostic.json").read_text()
    )
    calculator = Gfn1NativeCalculator(
        multipole=True,
        multipole_order=4,
        electronic_temperature=0.0,
        energy_tolerance=1e-13,
        charge_tolerance=1e-11,
        max_scc=1000,
    )
    result = calculator.calculate(
        [15, 1, 1, 1],
        [[0.1, 0.2, -0.1], [1.24, 0.15, 0.72], [-0.5, 1.22, 0.61], [-0.54, -0.91, 0.66]],
        compute_gradient=True,
    )
    assert result.converged
    assert abs(result.energy_hartree - reference["resolved_state"]["energy_hartree"]) < 1e-10
    expected = np.asarray(reference["resolved_state"]["richardson_gradient_hartree_per_bohr"]).reshape(4, 3)
    np.testing.assert_allclose(result.gradient_hartree_per_bohr, expected, atol=1e-9, rtol=1e-7)
