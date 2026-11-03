"""Installed pip CLI calls the same conditioned analytic Rust property backends."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

from gfn1_rs import Gfn1NativeCalculator
from gfn1_rs.native import FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM


def executable():
    directory = Path(os.environ.get("GFN1_CLI_BIN_DIR", Path(sys.executable).parent))
    return directory / ("gfn1-rs.exe" if sys.platform == "win32" else "gfn1-rs")


def invoke(*args, cwd=None):
    return subprocess.run([str(executable()), "properties", *map(str, args)],
                          cwd=cwd, capture_output=True, text=True, encoding="utf-8")


def test_installed_crystal_cli_catalog_and_rust_only_help():
    result = invoke("--list")
    assert result.returncode == 0, result.stderr
    catalog = json.loads(result.stdout)
    assert len(catalog) == 21
    names = {row["operation"] for row in catalog}
    assert {"electromechanical_response", "quasiharmonic_zero_pressure", "phonon_mesh"} <= names
    assert all(row["rust_backend"] for row in catalog)
    help_result = invoke("--help")
    assert help_result.returncode == 0 and "--request" in help_result.stdout


def test_installed_crystal_cli_relative_user_model_chain_and_native_agreement(tmp_path):
    work = tmp_path / "request directory"
    work.mkdir()
    model = work / "user weights.model"
    model.write_bytes((Path(__file__).resolve().parents[2] / "models/h2-radial-demo.model").read_bytes())
    numbers = [1, 1]
    positions = [[0, 0, 0], [0.8, 0.1, 0]]
    cell = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
    request = dict(schema="gfn1-rs-crystal-properties-v1",
        geometry=dict(numbers=numbers, positions=positions, cell=cell,
                      coordinate_units="angstrom", charge=0),
        electronic=dict(temperature_kelvin=300., energy_tolerance_hartree=1e-13,
                        charge_tolerance=1e-11),
        periodic=dict(kmesh=[1, 1, 1], gamma_centered=True, fold_time_reversal=False,
                      ao_cutoff_bohr=20.), ml_model_path=model.name,
        tasks=[dict(id="g", operation="ground_state"),
               dict(id="b", operation="band_structure", arguments=dict(sampling_mesh=[2, 1, 1])),
               dict(id="d", operation="electronic_dos", arguments=dict(
                   energies_hartree={"$ref": "b#/energies_hartree"}, weights={"$ref": "b#/integration_weights"},
                   grid_ev=[-40, -30, -20, -10, 0, 10, 20]))])
    source = work / "request.json"
    source.write_text(json.dumps(request), encoding="utf-8")
    output = tmp_path / "results.json"
    result = invoke("--request", source, "--output", output,
                    "--threads", "2", "--memory-mb", "256", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert "[gfn1-rs ML model]" in result.stderr
    report = json.loads(output.read_text())
    assert report["status"] == "completed"
    assert report["request_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert report["ml_model_identity"]["source_sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert report["conditions"]["electronic_temperature_kelvin"] == 300
    assert report["resources"]["threads"] == 2
    assert report["resources"]["workspace_budget_bytes"] == 256 * 1024 * 1024
    assert "ml_model_identity" not in report["tasks"][2]["provenance"]
    native = Gfn1NativeCalculator(electronic_temperature=300., energy_tolerance=1e-13,
        charge_tolerance=1e-11, pbc_ao_cutoff=20., ml_model_path=str(model), threads=2, memory_mb=256)
    expected = native.calculate_periodic(numbers, positions, cell, kgrid=(1, 1, 1),
        gamma_centered=True, compute_gradient=True, compute_stress=True)
    assert abs(report["tasks"][0]["result"]["energy_hartree"] - expected.energy_hartree) < 1e-11
    assert len(report["tasks"][0]["result"]["stress_hartree_per_bohr3"]) == 3
    assert len(report["tasks"][0]["result"]["forces_hartree_per_bohr"]) == 2
    ground = report["tasks"][0]["result"]
    native_forces = [[x / FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM for x in row]
                     for row in expected.forces_ev_per_angstrom]
    for actual, reference in zip(ground["gradient_hartree_per_bohr"], native_forces):
        assert max(abs(a+b) for a, b in zip(actual, reference)) < 1e-9
    for actual, reference in zip(ground["forces_hartree_per_bohr"], native_forces):
        assert max(abs(a-b) for a, b in zip(actual, reference)) < 1e-9
    for actual, reference in zip(ground["stress_hartree_per_bohr3"], expected.stress):
        assert max(abs(a-b) for a, b in zip(actual, reference)) < 1e-9
    assert report["tasks"][1]["result"]["integration_weights"] == [0.5, 0.5]


def test_installed_crystal_cli_preserves_failed_physics_and_nonzero_exit(tmp_path):
    source = tmp_path / "bad-physical-task.json"
    source.write_text(json.dumps(dict(schema="gfn1-rs-crystal-properties-v1", continue_on_error=True,
        tasks=[dict(id="unstable", operation="harmonic_thermodynamics", arguments=dict(
            wavenumbers_cm1=[[-100, 1000]], weights=[1], temperatures_kelvin=[300])),
        dict(id="stable", operation="harmonic_thermodynamics", arguments=dict(
            wavenumbers_cm1=[[100, 1000]], weights=[1], temperatures_kelvin=[300]))])))
    output = tmp_path / "partial.json"
    result = invoke("--request", source, "--output", output)
    assert result.returncode == 1
    assert "partial results" in result.stderr
    report = json.loads(output.read_text())
    assert report["status"] == "partial" and report["failed_tasks"] == 1
    assert report["tasks"][0]["status"] == "error"
    assert report["tasks"][1]["status"] == "completed"
    assert "negative" in report["tasks"][0]["error"] or "imaginary" in report["tasks"][0]["error"]


def test_sourced_sic_crystal_cli_matches_native_response(tmp_path):
    root = Path(__file__).resolve().parents[2]
    survey = json.loads((root / "tests/data/real-crystal-property-survey.json").read_text())
    case = next(c for c in survey["crystal_executions"] if c["execution_id"] == "sic")
    reference = case["operations"]["born_proper_piezo_dielectric_ir"]["result"]
    output = tmp_path / "sic-response.json"
    result = invoke("--request", root / "examples/crystal-sic-cod9008856-relaxed-request.json",
                    "--output", output, "--threads", "2", "--memory-mb", "2048")
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["status"] == "completed"
    assert report["conditions"]["electronic_temperature_kelvin"] == 300
    assert all(task["status"] == "completed" for task in report["tasks"])
    current = report["tasks"][0]["result"]
    paths = [("elastic", "clamped_stiffness_hartree_per_bohr3"),
             ("berry", "born_effective_charges"), ("berry", "clamped_piezoelectric_au"),
             ("electronic", "electronic_relative_dielectric"), ("relaxed_piezoelectric_au",),
             ("piezoelectric_strain_au",), ("static_dielectric_fixed_strain",),
             ("static_dielectric_free_stress",)]
    for path in paths:
        observed, expected = current, reference
        for key in path:
            observed, expected = observed[key], expected[key]
        assert observed is not None and expected is not None, path
        np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-8, err_msg=str(path))
    lo = report["tasks"][1]["result"]
    reference_lo = case["operations"]["lo_to_100"]["result"]
    for key in ("dynamical_matrix_hartree_per_bohr2_amu", "eigenvalues_hartree_per_bohr2_amu"):
        np.testing.assert_allclose(lo[key], reference_lo[key], rtol=0, atol=1e-10)
    observed = np.asarray(lo["wavenumbers_cm1"])
    expected = np.asarray(reference_lo["wavenumbers_cm1"])
    acoustic = (np.abs(observed) < 1e-3) & (np.abs(expected) < 1e-3)
    assert np.count_nonzero(acoustic) == 3
    np.testing.assert_allclose(observed[~acoustic], expected[~acoustic], rtol=0, atol=1e-6)
