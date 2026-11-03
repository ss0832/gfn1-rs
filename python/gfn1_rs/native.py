# SPDX-License-Identifier: GPL-3.0-or-later
"""Non-ASE native Python API for the Rust GFN1-xTB implementation.

This module is the home of the *native* calculator surface — the ``Gfn1NativeCalculator``
class compiled from Rust (PyO3, exposed as ``gfn1_rs._native``) plus its result types and the
``default_param_path`` helper. It does **not** depend on ASE.

``Gfn1NativeCalculator`` is the full, no-ASE property API: construct it once and call its
methods directly with atomic numbers + Cartesian positions. Every property the ASE
:class:`gfn1_rs.GFN1RSCalculator` exposes is available here (the ASE wrapper is a thin
convenience layer over this class).

Units: **this is an atomic-units API.** Everything it returns is in atomic units unless
the attribute/key name says otherwise — Hartree for energies, Hartree/bohr for gradients
and forces, Hartree/bohr**2 and Hartree/bohr**3 for the Hessian and cubic force constants,
Hartree/bohr**3 for the periodic stress (``sigma_ab = (1/V) dE/d eps_ab``), e*bohr for
dipoles, and bohr for the ``origin`` arguments. The two exceptions, both name-tagged, are
the ``*_ev`` / ``*_angstrom`` convenience fields on the result objects and the
spectroscopic outputs that carry a universal unit of their own (cm**-1 wavenumbers,
km/mol IR intensities, ppm NMR shieldings, 1e-30 J/T**2 magnetizabilities). Input
``positions`` / ``cell`` are read in the ``unit`` argument (``"angstrom"`` by default,
``"bohr"`` for atomic units).

Only :class:`gfn1_rs.ase.GFN1RSCalculator` converts to ASE's Angstrom / eV / e
convention, and it does so using ``ase.units`` exclusively. The ``HARTREE_TO_EV`` /
``BOHR_TO_ANGSTROM`` / ``FORCE_...`` / ``HESSIAN_...`` constants re-exported here are the
**native** (Rust) CODATA values used inside the engine's own ``*_ev`` fields; they differ
from ``ase.units`` at the 1e-8 level, so do not mix the two when reproducing an ASE
number.

Parameter resolution: ``param_path`` (a file path, or the builtin specs ``"builtin"`` /
``"builtin:si"``) > the ``GFN1_XTB_PARAM`` environment variable > the bundled official
GFN1-xTB parametrization (from grimme-lab/xtb, LGPL-3.0-or-later). Call
``calc.param_source()`` to see which parametrization is active.

Example
-------
>>> import gfn1_rs
>>> calc = gfn1_rs.Gfn1NativeCalculator()  # bundled GFN1-xTB parameters
>>> nums = [8, 1, 1]
>>> pos = [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]
>>> res = calc.calculate(numbers=nums, positions=pos, unit="angstrom")
>>> res.energy_hartree, res.dipole
>>> alpha = calc.polarizability(numbers=nums, positions=pos)        # static polarizability
>>> xi = calc.magnetizability(numbers=nums, positions=pos)          # 1e-30 J/T^2
>>> cm = calc.cotton_mouton(numbers=nums, positions=pos)            # d^2 alpha / dB^2

See ``docs/python-api.md`` for the full method list.
"""

import functools
import inspect
import logging
import os

from ._native import (  # noqa: F401
    ANGSTROM_TO_BOHR,
    BOHR_TO_ANGSTROM,
    FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
    HARTREE_TO_EV,
    HESSIAN_HARTREE_PER_BOHR2_TO_EV_PER_ANGSTROM2,
    MAX_FOURTH_DERIVATIVE_NDOF,
    CalculationResult,
    Gfn1NativeCalculator as _RawCalculator,
    _ResourceContext,
    OptimizationResult,
    roundtrip_param_file,
)
from .training import create_ml_model, train_ml_model


class Gfn1NativeCalculator:
    """Rust calculator with optional ``threads`` and workspace ``memory_mb``.

    Every property executes inside the same reusable Rust resource scope.
    Memory is an estimated calculation-workspace budget, not a process RSS cap.
    Other constructor and method arguments retain the native signatures.
    """

    def __init__(self, *args, threads=None, memory_mb=None,
                 cp_tolerance=None, cp_relative_tolerance=None, cp_max_iter=None, **kwargs):
        self._resources = _ResourceContext(threads=threads, memory_mb=memory_mb,
            cp_tolerance=cp_tolerance, cp_relative_tolerance=cp_relative_tolerance,cp_max_iter=cp_max_iter)
        self._raw = self._resources.run(lambda: _RawCalculator(*args, **kwargs))
        self._resource_info = {"threads": threads, "memory_budget_bytes": self._resources.info()[1]}
        if threads is not None or memory_mb is not None:
            message = f"gfn1-rs resources: threads={threads or 'rayon default'}, workspace_memory_mb={memory_mb or 'method default'}"
            logging.getLogger("gfn1_rs").info(message)
            if kwargs.get("log_console", False):
                print(message, file=__import__("sys").stderr)
        if any(value is not None for value in self._resources.cp_info()):
            message=f"gfn1-rs CP settings: {self.cp_settings()}"
            logging.getLogger("gfn1_rs").info(message)
            if kwargs.get("log_console",False):
                print(message,file=__import__("sys").stderr)

    def resource_limits(self):
        """Return the selected worker count and workspace budget in bytes."""
        return dict(self._resource_info)

    def cp_settings(self):
        """Return configured CP residual tolerances and iterative limit."""
        absolute,relative,iterations=self._resources.cp_info()
        return {"absolute_tolerance":absolute,"relative_tolerance":relative,"max_iterations":iterations}

    def __getattr__(self, name):
        return getattr(self._raw, name)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(dir(self._raw)))


def _resource_method(name, raw_method):
    @functools.wraps(raw_method)
    def scoped(self, *args, **kwargs):
        return self._resources.run(lambda: getattr(self._raw, name)(*args, **kwargs))
    try:
        scoped.__signature__ = inspect.signature(raw_method)
    except (TypeError, ValueError):
        pass
    return scoped


for _name in dir(_RawCalculator):
    if not _name.startswith("_") and callable(getattr(_RawCalculator, _name)):
        setattr(Gfn1NativeCalculator, _name, _resource_method(_name, getattr(_RawCalculator, _name)))

try:
    _signature = inspect.signature(_RawCalculator)
    Gfn1NativeCalculator.__signature__ = _signature.replace(parameters=[
        *_signature.parameters.values(),
        inspect.Parameter("threads", inspect.Parameter.KEYWORD_ONLY, default=None),
        inspect.Parameter("memory_mb", inspect.Parameter.KEYWORD_ONLY, default=None),
        inspect.Parameter("cp_tolerance", inspect.Parameter.KEYWORD_ONLY, default=None),
        inspect.Parameter("cp_relative_tolerance", inspect.Parameter.KEYWORD_ONLY, default=None),
        inspect.Parameter("cp_max_iter", inspect.Parameter.KEYWORD_ONLY, default=None),
    ])
except (TypeError, ValueError):
    pass


def default_param_path() -> str:
    """Return the ``GFN1_XTB_PARAM`` environment value, or ``"builtin"``.

    The returned string is a valid ``param_path`` argument for
    :class:`Gfn1NativeCalculator`. When the environment variable is unset the
    bundled official GFN1-xTB parametrization is used, so this never raises.
    """

    path = os.environ.get("GFN1_XTB_PARAM")
    return path if path else "builtin"


__all__ = [
    "ANGSTROM_TO_BOHR",
    "BOHR_TO_ANGSTROM",
    "FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM",
    "HARTREE_TO_EV",
    "HESSIAN_HARTREE_PER_BOHR2_TO_EV_PER_ANGSTROM2",
    "MAX_FOURTH_DERIVATIVE_NDOF",
    "CalculationResult",
    "Gfn1NativeCalculator",
    "OptimizationResult",
    "roundtrip_param_file",
    "default_param_path",
    "create_ml_model",
    "train_ml_model",
]
