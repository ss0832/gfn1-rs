"""Installed commands and analytic fractional-occupation public API contracts."""
import importlib.metadata
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from gfn1_rs import Gfn1NativeCalculator


@pytest.mark.parametrize("name", ["gfn1-rs", "gfn1-rs-train"])
def test_installed_cli_version_and_error_status(name):
    command = Path(sys.executable).parent / (name + (".exe" if sys.platform == "win32" else ""))
    version = subprocess.run([str(command), "--version"], capture_output=True, text=True, encoding="utf-8", check=True)
    assert version.stdout.strip() == f"{name} {importlib.metadata.version('gfn1-rs-python')}"
    bad = subprocess.run([str(command), "--unknown-option"], capture_output=True, text=True, encoding="utf-8")
    assert bad.returncode == 1 and "error:" in bad.stderr and "Traceback" not in bad.stderr


def test_installed_cli_calculation_and_training(tmp_path):
    bindir = Path(sys.executable).parent
    suffix = ".exe" if sys.platform == "win32" else ""
    xyz = tmp_path / "分子 with space.xyz"
    xyz.write_text("2\nH2\nH 0 0 0\nH .74 0 0\n")
    result = subprocess.run([str(bindir / ("gfn1-rs"+suffix)), str(xyz), "--gradient"], capture_output=True, text=True, encoding="utf-8", check=True)
    assert "gradient" in result.stdout and "total" in result.stdout
    output = tmp_path / "initial.model"
    subprocess.run([str(bindir / ("gfn1-rs-train"+suffix)), "new", "--species", "H,O", "--centers", "1.2,2.5", "--cutoff", "6", "--output", str(output)], capture_output=True, text=True, encoding="utf-8", check=True)
    assert output.is_file() and output.stat().st_size > 0


def test_fractional_cubic_vector_and_quartic_route_match_independent_differences():
    calc = Gfn1NativeCalculator(electronic_temperature=30000., energy_tolerance=1e-13, charge_tolerance=1e-11, max_scc=1000)
    nums = [1, 9]
    pos = np.array([[.1, -.2, .03], [1.8, .12, -.11]])
    v = np.array([.13, -.07, .21, -.11, .09, .03])
    tensor = np.asarray(calc.third_derivative(nums, pos.tolist(), unit="bohr"))
    vector = np.asarray(calc.third_derivative_vector(nums, pos.tolist(), v.tolist(), unit="bohr"))
    np.testing.assert_allclose(vector, np.einsum("abc,c->ab", tensor, v), atol=1e-10, rtol=1e-9)
    quartic = calc.fourth_derivative_directional(nums, pos.tolist(), v.tolist(), unit="bohr")
    def cubic(offset):
        return calc.third_derivative_finite_t_directional(nums, (pos+offset*v.reshape(2,3)).tolist(), v.tolist(), unit="bohr")
    h = 1e-3
    coarse = (cubic(h)-cubic(-h))/(2*h)
    fine = (cubic(h/2)-cubic(-h/2))/h
    assert abs(quartic-(4*fine-coarse)/3) < 1e-8
