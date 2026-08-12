# SPDX-License-Identifier: GPL-3.0-or-later
"""Python interface for the native Rust GFN1-xTB implementation.

Two layers, two unit conventions:

* :class:`gfn1_rs.Gfn1NativeCalculator` (and everything else re-exported here) is
  the **atomic-units** API — bohr, Hartree, Hartree/bohr, e*bohr. See
  :mod:`gfn1_rs.native`.
* :class:`gfn1_rs.GFN1RSCalculator` is the **ASE** layer — Angstrom, eV,
  eV/Angstrom, eV/Angstrom**3 (Voigt stress), e*Angstrom — converted with
  ``ase.units`` and nothing else. See :mod:`gfn1_rs.ase`.
"""

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
    MAX_FOURTH_DERIVATIVE_NDOF,
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
    "MAX_FOURTH_DERIVATIVE_NDOF",
    "CalculationResult",
    "Gfn1NativeCalculator",
    "OptimizationResult",
    "GFN1RSCalculator",
    "roundtrip_param_file",
    "default_param_path",
]
