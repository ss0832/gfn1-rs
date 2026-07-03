# SPDX-License-Identifier: GPL-3.0-or-later
"""Python interface for the native Rust GFN1-xTB implementation."""

from importlib.metadata import PackageNotFoundError, version as _dist_version

try:
    __version__ = _dist_version("gfn1-rs-python")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

# The non-ASE native API (Gfn1NativeCalculator + result types + helpers) lives in
# `native.py`; re-export it here so the top-level names are unchanged (backward compatible).
from .native import (  # noqa: F401
    ANGSTROM_TO_BOHR,
    BOHR_TO_ANGSTROM,
    FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
    HARTREE_TO_EV,
    HESSIAN_HARTREE_PER_BOHR2_TO_EV_PER_ANGSTROM2,
    CalculationResult,
    Gfn1NativeCalculator,
    OptimizationResult,
    default_param_path,
    roundtrip_param_file,
)


try:
    from .ase import GFN1RSCalculator  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency guard
    GFN1RSCalculator = None


__all__ = [
    "__version__",
    "ANGSTROM_TO_BOHR",
    "BOHR_TO_ANGSTROM",
    "FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM",
    "HARTREE_TO_EV",
    "HESSIAN_HARTREE_PER_BOHR2_TO_EV_PER_ANGSTROM2",
    "CalculationResult",
    "Gfn1NativeCalculator",
    "OptimizationResult",
    "GFN1RSCalculator",
    "roundtrip_param_file",
    "default_param_path",
]
