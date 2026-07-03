# SPDX-License-Identifier: GPL-3.0-or-later
"""Non-ASE native Python API for the Rust GFN1-xTB implementation.

This module is the home of the *native* calculator surface — the ``Gfn1NativeCalculator``
class compiled from Rust (PyO3, exposed as ``gfn1_rs._native``) plus its result types and the
``default_param_path`` helper. It does **not** depend on ASE.

``Gfn1NativeCalculator`` is the full, no-ASE property API: construct it once with a GFN1
parameter file and call its methods directly with atomic numbers + Cartesian positions. Every
property the ASE :class:`gfn1_rs.GFN1RSCalculator` exposes is available here (the ASE wrapper is
a thin convenience layer over this class).

Example
-------
>>> import gfn1_rs
>>> calc = gfn1_rs.Gfn1NativeCalculator(param_path=gfn1_rs.default_param_path())
>>> nums = [8, 1, 1]
>>> pos = [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]
>>> res = calc.calculate(numbers=nums, positions=pos, unit="angstrom")
>>> res.energy_hartree, res.dipole
>>> alpha = calc.polarizability(numbers=nums, positions=pos)        # static polarizability
>>> xi = calc.magnetizability(numbers=nums, positions=pos)          # 1e-30 J/T^2
>>> cm = calc.cotton_mouton(numbers=nums, positions=pos)            # d^2 alpha / dB^2

See ``docs/python-api.md`` for the full method list.
"""

import os

from ._native import (  # noqa: F401
    ANGSTROM_TO_BOHR,
    BOHR_TO_ANGSTROM,
    FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
    HARTREE_TO_EV,
    HESSIAN_HARTREE_PER_BOHR2_TO_EV_PER_ANGSTROM2,
    CalculationResult,
    Gfn1NativeCalculator,
    OptimizationResult,
    roundtrip_param_file,
)


def default_param_path() -> str:
    """Return the ``GFN1_XTB_PARAM`` environment value (or raise if unset)."""

    path = os.environ.get("GFN1_XTB_PARAM")
    if not path:
        raise ValueError("set GFN1_XTB_PARAM or pass param_path explicitly")
    return path


__all__ = [
    "ANGSTROM_TO_BOHR",
    "BOHR_TO_ANGSTROM",
    "FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM",
    "HARTREE_TO_EV",
    "HESSIAN_HARTREE_PER_BOHR2_TO_EV_PER_ANGSTROM2",
    "CalculationResult",
    "Gfn1NativeCalculator",
    "OptimizationResult",
    "roundtrip_param_file",
    "default_param_path",
]
