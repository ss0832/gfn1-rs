"""End-to-end public training/loading recipes, without external executables."""
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import numpy as np
from ase import Atoms
from ase.units import Hartree
from gfn1_rs import GFN1RSCalculator
from gfn1_rs.native import Gfn1NativeCalculator, create_ml_model, train_ml_model


def test_user_training_identity_cli_and_ase(tmp_path, capfd, caplog):
    root = Path(__file__).resolve().parents[2]
    initial = tmp_path / "initial.model"
    trained = tmp_path / "user weights.model"
    report_path = tmp_path / "report.json"
    create_ml_model(initial, ["H"], [1.2, 2.5])
    report = train_ml_model(initial, root / "models/h2-fci-reference.json", trained,
                            report_path, epochs=5, learning_rate=.0007,
                            train_correction=True, gradient_weight=0,
                            dipole_weight=0, relative_energy_weight=0)
    assert report["status"] == "complete"
    assert trained.read_bytes() != initial.read_bytes()
    source_sha = hashlib.sha256(trained.read_bytes()).hexdigest()
    logfile = tmp_path / "native.jsonl"
    calc = Gfn1NativeCalculator(ml_model_path=str(trained), log_file=str(logfile),
                               log_console=True, electronic_temperature=0.)
    info = calc.ml_model_info()
    assert Path(info["source"]).samefile(trained)
    assert info["source_sha256"] == source_sha
    assert info["source_sha256"] == info["canonical_sha256"]
    assert report["dataset_sha256"] in info["training_provenance"]
    result = calc.calculate([1, 1], [[0, 0, 0], [.74, 0, 0]])
    assert result.ml_model_sha256 == info["canonical_sha256"]
    records = [json.loads(line) for line in logfile.read_text().splitlines()]
    assert any(r["event"] == "ml_model_loaded" and r["data"]["source_sha256"] == source_sha for r in records)
    assert source_sha in capfd.readouterr().err
    caplog.set_level(logging.INFO, logger="gfn1_rs")
    atoms = Atoms("H2", positions=[[0, 0, 0], [.74, 0, 0]])
    atoms.calc = GFN1RSCalculator(ml_model_path=str(trained), electronic_temperature=0.)
    np.testing.assert_allclose(atoms.get_potential_energy(), result.energy_hartree*Hartree, atol=1e-10)
    assert source_sha in caplog.text and "user weights.model" in caplog.text
    assert atoms.calc.results["ml_model_sha256"] == source_sha
    xyz = tmp_path / "h2.xyz"
    xyz.write_text("2\nH2\nH 0 0 0\nH .74 0 0\n")
    cli = Path(sys.executable).parent / ("gfn1-rs.exe" if sys.platform == "win32" else "gfn1-rs")
    output = subprocess.run([str(cli), str(xyz), "--ml-model", str(trained)],
                            capture_output=True, text=True, encoding="utf-8", check=True).stdout
    assert source_sha in output and "user weights.model" in output
    assert Gfn1NativeCalculator().ml_model_info() is None
