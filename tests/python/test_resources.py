"""Real calculations through scoped Rust, native/ASE and pip command resources."""
import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from ase import Atoms
from gfn1_rs import Gfn1NativeCalculator, GFN1RSCalculator
from gfn1_rs._native import Gfn1NativeCalculator as RawCalculator


NUMS = [8, 1, 1]
POS = [[0, 0, 0], [.757, .586, 0], [-.757, .586, 0]]


def test_native_scopes_preserve_signatures_forces_and_log_actual_workers(tmp_path):
    calculators = [Gfn1NativeCalculator(threads=n, memory_mb=128,
                   energy_tolerance=1e-12, charge_tolerance=1e-10,
                   log_file=str(tmp_path / f"workers-{n}.jsonl")) for n in (1, 2)]
    def calculate(calc):
        return calc.calculate(NUMS, POS, compute_gradient=True)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(calculate, calculators))
    assert abs(results[0].energy_hartree-results[1].energy_hartree) < 1e-13
    np.testing.assert_allclose(results[0].gradient_hartree_per_bohr,
                               results[1].gradient_hartree_per_bohr, atol=1e-12, rtol=0)
    for n, calc in zip((1, 2), calculators):
        assert calc.resource_limits() == {"threads": n, "memory_budget_bytes": 128*1024**2}
        entries = [json.loads(line) for line in (tmp_path / f"workers-{n}.jsonl").read_text().splitlines()]
        assert any(entry.get("data", {}).get("configuration", {}).get("resources", {}).get("threads") == n for entry in entries)
        assert inspect.signature(calc.calculate) == inspect.signature(RawCalculator().calculate)
        assert inspect.signature(calc.stda_xtb) == inspect.signature(RawCalculator().stda_xtb)
    assert "threads" in inspect.signature(Gfn1NativeCalculator).parameters
    assert "memory_mb" in inspect.signature(Gfn1NativeCalculator).parameters


def test_ase_resource_forwarding_and_changed_context():
    atoms = Atoms("OH2", positions=POS)
    atoms.calc = GFN1RSCalculator(threads=1, memory_mb=128)
    initial = atoms.get_potential_energy()
    assert atoms.calc._native.resource_limits()["threads"] == 1
    atoms.calc.set(threads=2)
    assert atoms.calc._native.resource_limits()["threads"] == 2
    assert abs(atoms.get_potential_energy()-initial) < 1e-10


def test_workspace_rejection_precedes_scf_and_explicit_budget_cannot_raise_it():
    # Ten separated H atoms give a full FC3 polarization plan larger than 1 MiB.
    # max_scc=0 makes accidental progression into SCF observable.
    calc = Gfn1NativeCalculator(threads=1, memory_mb=1, max_scc=0)
    with pytest.raises(ValueError, match="budget"):
        calc.third_derivative([1]*10, [[i*2., 0., 0.] for i in range(10)])
    with pytest.raises(ValueError, match="budget"):
        calc.phonons_commensurate([1, 1], [[0,0,0],[0,0,.75]],
            [[5.,0,0],[0,5.,0],[0,0,5.]], (4,4,4), [(0.,0.,0.)],
            memory_budget_bytes=2**30, max_atoms=256)
    for options in ({"threads":0}, {"memory_mb":0}):
        with pytest.raises(ValueError):
            Gfn1NativeCalculator(**options)


@pytest.mark.parametrize("name", ["gfn1-rs", "gfn1-rs-train"])
def test_pip_cli_common_resource_options(name, tmp_path):
    executable = Path(sys.executable).parent / (name + (".exe" if sys.platform == "win32" else ""))
    if name == "gfn1-rs":
        xyz = tmp_path / "h2.xyz"
        xyz.write_text("2\nH2\nH 0 0 0\nH 0 0 .75\n")
        arguments = [str(xyz)]
    else:
        arguments = ["new", "--species", "H", "--centers", "1.2,2.5", "--cutoff", "6", "--output", str(tmp_path/"initial.model")]
    completed = subprocess.run([str(executable), *arguments, "--threads", "2", "--memory-mb", "128"], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "threads=2" in completed.stderr and "workspace_budget_bytes=134217728" in completed.stderr
    invalid = subprocess.run([str(executable), "--threads", "0", "--version"], capture_output=True, text=True)
    assert invalid.returncode == 1 and "positive" in invalid.stderr
