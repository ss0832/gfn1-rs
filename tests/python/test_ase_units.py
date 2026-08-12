# SPDX-License-Identifier: GPL-3.0-or-later
"""ASE unit-convention contract for :class:`gfn1_rs.ase.GFN1RSCalculator`.

The rule under test: the ASE calculator layer speaks **ASE units** (Angstrom /
eV / e) and derives every conversion from ``ase.units`` (``Bohr``, ``Hartree``,
``invcm``), while the native PyO3 API (``gfn1_rs.Gfn1NativeCalculator``) keeps
**atomic units** (bohr / Hartree / e*bohr).

Every assertion here is therefore of the form ``ase_value == native_value *
exact_ase_units_factor`` -- no tolerance on the physics, only on floating-point
round-off, so the test fails if any conversion is dropped, doubled, or taken
from a different CODATA set than ``ase.units``.

The tests that need the compiled extension are skipped when it is unavailable;
:class:`TestConversionPolicyPureNumpy` exercises the pure-Python conversion
logic against a mocked native layer and always runs.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ase = pytest.importorskip("ase")

from ase import Atoms  # noqa: E402
from ase.units import Bohr, Hartree, invcm  # noqa: E402

try:  # the compiled extension is optional for the pure-Python tests below
    import gfn1_rs
    from gfn1_rs.ase import GFN1RSCalculator

    _NATIVE_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    gfn1_rs = None
    GFN1RSCalculator = None
    _NATIVE_IMPORT_ERROR = exc

requires_native = pytest.mark.skipif(
    GFN1RSCalculator is None,
    reason=f"gfn1_rs native extension unavailable: {_NATIVE_IMPORT_ERROR}",
)

# Exact expected conversion factors, built only from `ase.units`.
E = Hartree                     # Hartree      -> eV
F = Hartree / Bohr              # Hartree/bohr -> eV/Angstrom
H2 = Hartree / Bohr**2          # Hartree/bohr^2 -> eV/Angstrom^2
H3 = Hartree / Bohr**3          # Hartree/bohr^3 -> eV/Angstrom^3 (stress & cubic)
D = Bohr                        # e*bohr       -> e*Angstrom
POL = Bohr**2 / Hartree         # e^2 bohr^2/Ha -> e^2 Angstrom^2/eV

# Relative tolerance: pure float round-off from a multiply (and, for the periodic
# forces, one extra divide), never a physical tolerance.
RTOL = 1.0e-12


def water() -> Atoms:
    return Atoms(
        "OH2",
        positions=[
            [0.0, 0.0, 0.117],
            [0.0, 0.757, -0.469],
            [0.0, -0.757, -0.469],
        ],
    )


def diamond() -> Atoms:
    """The conventional diamond cell used by the Rust PBC suite, with one atom
    nudged off its site so both the stress and the forces are non-zero (the ideal
    lattice is a symmetry equilibrium, which would make the force check vacuous).
    """

    a = 3.567
    q = a / 4.0
    positions = [
        [0.0, 0.0, 0.0],
        [q, q, q],
        [0.0, 2 * q, 2 * q],
        [q, 3 * q, 3 * q],
        [2 * q, 0.0, 2 * q],
        [3 * q, q, 3 * q],
        [2 * q, 2 * q, 0.0],
        [3 * q, 3 * q, q],
    ]
    positions[0] = [0.05, 0.03, -0.04]
    return Atoms("C8", positions=positions, cell=[a, a, a], pbc=True)


@pytest.fixture(scope="module")
def calc():
    return GFN1RSCalculator()


@pytest.fixture(scope="module")
def native():
    return gfn1_rs.Gfn1NativeCalculator()


def numbers_positions(atoms: Atoms):
    return (
        atoms.get_atomic_numbers().astype(np.uint8).tolist(),
        np.asarray(atoms.get_positions(), dtype=float).tolist(),
    )


# ---------------------------------------------------------------------------
# The conversion table itself -- guards against a hand-typed literal creeping
# back in, or the module silently reverting to the native CODATA constants.
# ---------------------------------------------------------------------------
@requires_native
class TestConversionTable:
    def test_module_constants_come_from_ase_units(self):
        from gfn1_rs import ase as ase_mod

        assert ase_mod._ENERGY == Hartree
        assert ase_mod._LENGTH == Bohr
        assert ase_mod._FORCE == Hartree / Bohr
        assert ase_mod._HESSIAN == Hartree / Bohr**2
        assert ase_mod._THIRD == Hartree / Bohr**3
        assert ase_mod._FOURTH == Hartree / Bohr**4
        assert ase_mod._DIPOLE == Bohr
        assert ase_mod._POLARIZABILITY == Bohr**2 / Hartree
        assert ase_mod._WAVENUMBER == invcm

    def test_ase_constants_differ_from_native_constants(self):
        """The two CODATA sets really are different, so the test above has teeth."""

        from gfn1_rs._native import BOHR_TO_ANGSTROM, HARTREE_TO_EV

        assert HARTREE_TO_EV != Hartree
        assert BOHR_TO_ANGSTROM != Bohr


# ---------------------------------------------------------------------------
# Non-periodic single point: energy / forces / dipole / charges.
# ---------------------------------------------------------------------------
@requires_native
class TestMolecularSinglePoint:
    @pytest.fixture(scope="class")
    def pair(self, calc, native):
        atoms = water()
        atoms.calc = calc
        nums, pos = numbers_positions(atoms)
        ref = native.calculate(
            numbers=nums, positions=pos, unit="angstrom", compute_gradient=True
        )
        atoms.get_forces()  # populate every key of calc.results, gradient included
        return atoms, ref

    def test_energy_is_hartree_times_ase_hartree(self, pair):
        atoms, ref = pair
        assert atoms.get_potential_energy() == pytest.approx(
            ref.energy_hartree * E, rel=RTOL
        )

    def test_forces_are_minus_gradient_times_hartree_over_bohr(self, pair):
        atoms, ref = pair
        gradient_au = np.asarray(ref.gradient_hartree_per_bohr, dtype=float)
        np.testing.assert_allclose(
            atoms.get_forces(), -gradient_au * F, rtol=RTOL, atol=0.0
        )

    def test_dipole_is_e_bohr_times_bohr(self, pair):
        atoms, ref = pair
        np.testing.assert_allclose(
            atoms.get_dipole_moment(),
            np.asarray(ref.dipole, dtype=float) * D,
            rtol=RTOL,
            atol=0.0,
        )

    def test_charges_are_unconverted_electrons(self, pair):
        atoms, ref = pair
        np.testing.assert_allclose(
            atoms.get_charges(), np.asarray(ref.charges, dtype=float), rtol=RTOL
        )

    def test_energy_terms_ev_are_hartree_terms_times_ase_hartree(self, pair):
        atoms, _ = pair
        results = atoms.calc.results
        hartree = results["native_energy_terms_hartree"]
        ev = results["native_energy_terms_ev"]
        assert set(hartree) == set(ev)
        for key, value in hartree.items():
            assert ev[key] == pytest.approx(value * E, rel=RTOL)

    def test_native_passthrough_keys_stay_atomic(self, pair):
        atoms, ref = pair
        results = atoms.calc.results
        np.testing.assert_allclose(
            results["native_dipole_au"], np.asarray(ref.dipole, dtype=float)
        )
        np.testing.assert_allclose(
            results["native_forces_hartree_per_bohr"],
            -np.asarray(ref.gradient_hartree_per_bohr, dtype=float),
            rtol=RTOL,
        )
        # ...and the ASE key is exactly that, converted.
        np.testing.assert_allclose(
            results["forces"],
            results["native_forces_hartree_per_bohr"] * F,
            rtol=RTOL,
            atol=0.0,
        )

    def test_energy_does_not_use_the_native_hartree_constant(self, pair):
        """A regression guard: `result.energy_ev` is scaled by the Rust CODATA
        constant, which is NOT what the ASE layer must return."""

        atoms, ref = pair
        energy = atoms.get_potential_energy()
        assert energy == pytest.approx(ref.energy_hartree * E, rel=RTOL)
        if abs(ref.energy_hartree) > 1.0:
            assert energy != pytest.approx(ref.energy_ev, rel=1.0e-11)


# ---------------------------------------------------------------------------
# Periodic single point: stress (Voigt, ASE sign) and periodic forces.
# ---------------------------------------------------------------------------
@requires_native
class TestPeriodicSinglePoint:
    @pytest.fixture(scope="class")
    def pair(self, calc, native):
        atoms = diamond()
        atoms.calc = calc
        nums, pos = numbers_positions(atoms)
        ref = native.calculate_periodic(
            numbers=nums,
            positions=pos,
            cell=np.asarray(atoms.get_cell(), dtype=float).tolist(),
            pbc=(True, True, True),
            kgrid=None,
            unit="angstrom",
            compute_gradient=True,
            compute_stress=True,
        )
        atoms.get_stress()
        return atoms, ref

    def test_periodic_energy(self, pair):
        atoms, ref = pair
        assert atoms.get_potential_energy() == pytest.approx(
            ref.energy_hartree * E, rel=RTOL
        )

    def test_stress_is_voigt_of_native_tensor_times_hartree_per_bohr3(self, pair):
        atoms, ref = pair
        s = np.asarray(ref.stress, dtype=float) * H3
        expected = np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])
        np.testing.assert_allclose(atoms.get_stress(), expected, rtol=RTOL, atol=0.0)

    def test_stress_voigt_ordering_matches_ase(self, pair):
        """ASE Voigt order is (xx, yy, zz, yz, xz, xy) -- check against ASE itself.

        ASE's own ``voigt=False`` expansion is the reference for the ordering; each
        component is then matched to the converted native tensor entry it must have
        come from. (The comparison is component-wise because ASE's expansion
        symmetrizes, while the native tensor carries ~1e-11 relative asymmetry.)
        """

        atoms, ref = pair
        voigt = atoms.get_stress()
        full = atoms.get_stress(voigt=False)
        expanded = np.array(
            [
                [voigt[0], voigt[5], voigt[4]],
                [voigt[5], voigt[1], voigt[3]],
                [voigt[4], voigt[3], voigt[2]],
            ]
        )
        np.testing.assert_allclose(full, expanded, rtol=RTOL, atol=0.0)

        raw = np.asarray(ref.stress, dtype=float) * H3
        for i, j in ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)):
            assert full[i, j] == pytest.approx(raw[i, j], rel=RTOL)

    def test_periodic_forces_round_trip_to_ase_units(self, pair):
        from gfn1_rs._native import FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM as NF

        atoms, ref = pair
        forces_au = np.asarray(ref.forces_ev_per_angstrom, dtype=float) / NF
        np.testing.assert_allclose(
            atoms.get_forces(), forces_au * F, rtol=1.0e-11, atol=0.0
        )

    def test_native_stress_passthrough_is_atomic(self, pair):
        atoms, ref = pair
        np.testing.assert_allclose(
            atoms.calc.results["native_stress_hartree_per_bohr3"],
            np.asarray(ref.stress, dtype=float),
        )


# ---------------------------------------------------------------------------
# Higher derivatives and response properties.
# ---------------------------------------------------------------------------
@requires_native
class TestDerivedProperties:
    @pytest.fixture(scope="class")
    def atoms(self, calc):
        atoms = water()
        atoms.calc = calc
        return atoms

    def test_hessian_is_ev_per_angstrom_squared(self, atoms, native):
        nums, pos = numbers_positions(atoms)
        raw = np.asarray(
            native.hessian(numbers=nums, positions=pos, unit="angstrom"), dtype=float
        )
        np.testing.assert_allclose(
            atoms.calc.get_hessian(atoms), raw * H2, rtol=RTOL, atol=0.0
        )

    def test_third_derivative_vector_is_ev_per_angstrom_cubed(self, atoms, native):
        nums, pos = numbers_positions(atoms)
        direction = np.zeros(3 * len(atoms))
        direction[2] = 1.0
        raw = np.asarray(
            native.third_derivative_vector(
                numbers=nums,
                positions=pos,
                direction=direction.tolist(),
                unit="angstrom",
            ),
            dtype=float,
        )
        np.testing.assert_allclose(
            atoms.calc.get_third_derivative_vector(direction, atoms),
            raw * H3,
            rtol=RTOL,
            atol=0.0,
        )

    def test_polarizability_is_e2_angstrom2_per_ev(self, atoms, native):
        nums, pos = numbers_positions(atoms)
        raw = native.polarizability(numbers=nums, positions=pos, unit="angstrom")
        out = atoms.calc.get_polarizability(atoms)
        np.testing.assert_allclose(
            out["tensor"],
            np.asarray(raw["tensor"], dtype=float) * POL,
            rtol=RTOL,
            atol=0.0,
        )
        assert out["isotropic"] == pytest.approx(raw["isotropic"] * POL, rel=RTOL)
        assert out["anisotropy"] == pytest.approx(raw["anisotropy"] * POL, rel=RTOL)
        np.testing.assert_allclose(out["tensor_au"], raw["tensor"])

    def test_dipole_derivatives_dipole_is_converted_tensor_is_not(self, atoms, native):
        nums, pos = numbers_positions(atoms)
        raw = native.dipole_derivatives(numbers=nums, positions=pos, unit="angstrom")
        out = atoms.calc.get_dipole_derivatives(atoms)
        np.testing.assert_allclose(
            out["dipole"], np.asarray(raw["dipole"], dtype=float) * D, rtol=RTOL
        )
        # dmu/dR is e*bohr per bohr == e*Angstrom per Angstrom: invariant.
        np.testing.assert_allclose(
            out["ddipole_dr"], np.asarray(raw["ddipole_dr"], dtype=float)
        )

    def test_vibrational_energies_are_wavenumbers_times_invcm(self, atoms):
        out = atoms.calc.get_vibrational_frequencies(atoms)
        np.testing.assert_allclose(
            out["energies_ev"],
            np.asarray(out["wavenumbers"], dtype=float) * invcm,
            rtol=RTOL,
            atol=0.0,
        )

    def test_magnetic_energy_is_ev(self, atoms, native):
        nums, pos = numbers_positions(atoms)
        b = (0.0, 0.0, 0.01)
        raw = float(
            native.magnetic_energy(
                numbers=nums, positions=pos, b_field=b, unit="angstrom"
            )
        )
        assert atoms.calc.get_magnetic_energy(b, atoms) == pytest.approx(
            raw * E, rel=RTOL
        )

    def test_magnetic_forces_dict_follows_the_suffix_rule(self, atoms, native):
        nums, pos = numbers_positions(atoms)
        b = (0.0, 0.0, 0.01)
        raw = native.magnetic_forces(
            numbers=nums, positions=pos, b_field=b, unit="angstrom"
        )
        out = atoms.calc.get_magnetic_forces(b, atoms)
        assert out["energy"] == pytest.approx(raw["energy_hartree"] * E, rel=RTOL)
        assert out["energy_hartree"] == pytest.approx(raw["energy_hartree"])
        np.testing.assert_allclose(
            out["forces"], np.asarray(raw["forces"], dtype=float) * F, rtol=RTOL
        )
        np.testing.assert_allclose(
            out["forces_hartree_per_bohr"], np.asarray(raw["forces"], dtype=float)
        )
        np.testing.assert_allclose(
            out["gradient"], np.asarray(raw["gradient"], dtype=float) * F, rtol=RTOL
        )

    def test_gauge_origin_is_taken_in_angstrom(self, atoms, native):
        """``origin`` is a length, so the ASE layer must hand the native (bohr) API
        the Angstrom value divided by Bohr.

        Checked on the angular-momentum integrals ``L = (r - O) x p``, which are
        strongly origin dependent (the molecular dipole and dmu/dR of a *neutral*
        system are origin invariant, so they cannot detect the bug).
        """

        nums, pos = numbers_positions(atoms)
        origin_angstrom = (0.3, -0.2, 0.5)
        origin_bohr = tuple(v / Bohr for v in origin_angstrom)

        expected = np.asarray(
            native.angular_momentum(
                numbers=nums, positions=pos, unit="angstrom", origin=origin_bohr
            ),
            dtype=float,
        )
        got = atoms.calc.get_angular_momentum(atoms, origin=origin_angstrom)
        np.testing.assert_allclose(got, expected, rtol=RTOL, atol=0.0)

        # Not vacuous: feeding the raw Angstrom numbers as bohr gives something else.
        unconverted = np.asarray(
            native.angular_momentum(
                numbers=nums, positions=pos, unit="angstrom", origin=origin_angstrom
            ),
            dtype=float,
        )
        assert not np.allclose(expected, unconverted)


# ---------------------------------------------------------------------------
# Pure-Python conversion logic, with the native layer mocked out. These run even
# when the compiled extension is missing.
# ---------------------------------------------------------------------------
class _FakeResult:
    """Duck-types ``gfn1_rs._native.CalculationResult``."""

    def __init__(self, *, energy_hartree, gradient=None, forces_ev=None, stress=None):
        self.energy_hartree = energy_hartree
        self.energy_ev = energy_hartree * 27.211386245988  # native CODATA on purpose
        self.gradient_hartree_per_bohr = gradient
        self.forces_ev_per_angstrom = forces_ev
        self.charges = [0.25, -0.25]
        self.dipole = (0.1, -0.2, 0.35)
        self.stress = stress
        self.iterations = 7
        self.converged = True

    def energy_terms_hartree(self):
        return {"total_internal": self.energy_hartree, "repulsion": 0.125}

    def energy_terms_ev(self):
        return {k: v * 27.211386245988 for k, v in self.energy_terms_hartree().items()}


ASE_SOURCE = Path(__file__).resolve().parents[2] / "python" / "gfn1_rs" / "ase.py"


@pytest.fixture(scope="module")
def ase_mod():
    """Load ``python/gfn1_rs/ase.py`` against a **stubbed** ``gfn1_rs._native``.

    This exercises the pure-Python conversion logic of the file in the working
    tree with no compiled extension involved, so the unit contract stays testable
    even where the wheel cannot be built. The stub deliberately advertises the
    *native* CODATA constants, so any place where the module reached for them
    instead of ``ase.units`` would show up immediately.
    """

    pkg = types.ModuleType("gfn1_rs")
    pkg.__path__ = []  # mark it as a package so relative imports resolve

    native = types.ModuleType("gfn1_rs._native")
    native.HARTREE_TO_EV = 27.211386245988
    native.ANGSTROM_TO_BOHR = 1.8897261246257702
    native.BOHR_TO_ANGSTROM = 1.0 / native.ANGSTROM_TO_BOHR
    native.FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = (
        native.HARTREE_TO_EV / native.BOHR_TO_ANGSTROM
    )
    native.HESSIAN_HARTREE_PER_BOHR2_TO_EV_PER_ANGSTROM2 = (
        native.HARTREE_TO_EV / native.BOHR_TO_ANGSTROM**2
    )
    native.Gfn1NativeCalculator = object
    native.CalculationResult = _FakeResult
    native.OptimizationResult = object
    native.roundtrip_param_file = lambda *a, **k: None

    native_py = types.ModuleType("gfn1_rs.native")
    native_py.Gfn1NativeCalculator = object
    native_py.default_param_path = lambda: "builtin"

    keys = ("gfn1_rs", "gfn1_rs._native", "gfn1_rs.native", "gfn1_rs.ase")
    saved = {key: sys.modules.get(key) for key in keys}
    sys.modules["gfn1_rs"] = pkg
    sys.modules["gfn1_rs._native"] = native
    sys.modules["gfn1_rs.native"] = native_py
    sys.modules.pop("gfn1_rs.ase", None)
    try:
        spec = importlib.util.spec_from_file_location("gfn1_rs._ase_under_test", ASE_SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


class TestConversionPolicyPureNumpy:
    """Conversion helpers, verified without touching the Rust engine."""

    def test_constants_are_exactly_the_ase_units_ones(self, ase_mod):
        assert ase_mod._ENERGY == Hartree
        assert ase_mod._LENGTH == Bohr
        assert ase_mod._FORCE == Hartree / Bohr
        assert ase_mod._HESSIAN == Hartree / Bohr**2
        assert ase_mod._THIRD == Hartree / Bohr**3
        assert ase_mod._FOURTH == Hartree / Bohr**4
        assert ase_mod._DIPOLE == Bohr
        assert ase_mod._POLARIZABILITY == Bohr**2 / Hartree
        assert ase_mod._WAVENUMBER == invcm

    def test_no_hand_typed_conversion_literals_in_the_module(self):
        """Guard the 'ase.units is the single source' rule at the source level."""

        source = ASE_SOURCE.read_text(encoding="utf-8")
        code = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        )
        for literal in ("27.211", "0.52917", "1.88972", "1822.88"):
            assert literal not in code, f"hand-typed conversion literal {literal!r}"

    def test_forces_helper_prefers_the_analytic_gradient(self, ase_mod):
        gradient = [[0.1, -0.2, 0.3], [-0.1, 0.2, -0.3]]
        result = _FakeResult(energy_hartree=-5.0, gradient=gradient)
        np.testing.assert_allclose(
            ase_mod._forces_hartree_per_bohr(result), -np.asarray(gradient)
        )

    def test_forces_helper_unscales_the_periodic_ev_forces(self, ase_mod):
        forces_au = np.array([[0.01, -0.02, 0.03], [-0.01, 0.02, -0.03]])
        native_factor = ase_mod._NATIVE_FORCE_FACTOR
        result = _FakeResult(
            energy_hartree=-5.0, forces_ev=(forces_au * native_factor).tolist()
        )
        np.testing.assert_allclose(
            ase_mod._forces_hartree_per_bohr(result), forces_au, rtol=1.0e-13
        )

    def test_forces_helper_returns_none_without_a_gradient(self, ase_mod):
        assert ase_mod._forces_hartree_per_bohr(_FakeResult(energy_hartree=-5.0)) is None

    def test_origin_helper_converts_angstrom_to_bohr(self, ase_mod):
        assert ase_mod._origin_bohr(None) is None
        np.testing.assert_allclose(
            ase_mod._origin_bohr((1.0, -2.0, 3.5)),
            np.array([1.0, -2.0, 3.5]) / Bohr,
            rtol=1.0e-14,
        )

    def test_gradient_dict_helper_applies_the_suffix_rule(self, ase_mod):
        raw = {
            "energy_hartree": -3.5,
            "gradient": [[1.0, 2.0, 3.0]],
            "forces": [[-1.0, -2.0, -3.0]],
        }
        out = ase_mod._to_ase_gradient_dict(
            dict(raw), energy_keys=(("energy_hartree", "energy"),)
        )
        assert out["energy"] == pytest.approx(-3.5 * Hartree, rel=1.0e-14)
        assert out["energy_hartree"] == -3.5
        np.testing.assert_allclose(
            out["gradient"], np.asarray(raw["gradient"]) * (Hartree / Bohr)
        )
        np.testing.assert_allclose(
            out["gradient_hartree_per_bohr"], np.asarray(raw["gradient"])
        )
        np.testing.assert_allclose(
            out["forces"], np.asarray(raw["forces"]) * (Hartree / Bohr)
        )
        np.testing.assert_allclose(
            out["forces_hartree_per_bohr"], np.asarray(raw["forces"])
        )

    def test_gradient_dict_helper_is_a_noop_without_those_keys(self, ase_mod):
        out = ase_mod._to_ase_gradient_dict({"wavenumbers": [1.0, 2.0]})
        assert out == {"wavenumbers": [1.0, 2.0]}

    def test_dipole_property_is_advertised(self, ase_mod):
        assert "dipole" in ase_mod.GFN1RSCalculator.implemented_properties
        for name in ("energy", "forces", "stress", "charges"):
            assert name in ase_mod.GFN1RSCalculator.implemented_properties
