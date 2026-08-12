# SPDX-License-Identifier: GPL-3.0-or-later
"""Smoke tests for the v0.5.0 derivative stack exposed through the Python API.

Four things are checked, one per feature group the v0.5.0 bindings added:

* **FC3** -- the dense cubic force constants agree with the memory-lean Vector
  mode (closed form, tight), the new semi-numerical Dense mode reproduces the
  closed form to finite-difference accuracy, and its Block mode is bit-for-bit
  the corresponding sub-block;
* **finite-temperature FC3** -- the occupation-agnostic directional route equals
  the T = 0 adjoint assembly on water, and on a strongly **Fermi-smeared**
  fixture (non-equilibrium formaldehyde at 10000 K, where the T = 0 orbital
  algebra does not apply at all) it reproduces the central finite difference of
  the smeared analytic Hessian;
* **FC4** -- the analytic directional quartic matches its semi-numerical
  reference at one step, the Block mode reconstructs the same directional value,
  and the expanded-tensor size cap is enforced;
* **PBC / Grueneisen** -- diamond's mode Grueneisen parameter comes out at the
  documented ``0.905``, and the periodic third derivative's Vector mode is an
  exact contraction of its Dense mode.

Plus the ASE unit contract of the three new calculator methods: ``eV/Angstrom^3``
for the finite-temperature cubic, ``eV/Angstrom^4`` for the quartic, and
**dimensionless** Grueneisen parameters with ``cm^-1`` frequencies.

The tests that need the compiled extension are skipped when it is unavailable.
Timings on a laptop-class machine: the whole file is a few minutes, dominated by
the periodic Hessian sweeps (the two diamond tests) -- the real-space cutoffs
there are reduced from the library defaults purely for speed, exactly as the Rust
integration gates in ``tests/pbc_third_derivative.rs`` do.
"""

from __future__ import annotations

import numpy as np
import pytest

ase = pytest.importorskip("ase")

from ase import Atoms  # noqa: E402
from ase.units import Bohr, Hartree  # noqa: E402

try:  # the compiled extension is optional
    import gfn1_rs
    from gfn1_rs.ase import GFN1RSCalculator

    _NATIVE_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    gfn1_rs = None
    GFN1RSCalculator = None
    _NATIVE_IMPORT_ERROR = exc

pytestmark = pytest.mark.skipif(
    GFN1RSCalculator is None,
    reason=f"gfn1_rs native extension unavailable: {_NATIVE_IMPORT_ERROR}",
)

# Exact ASE conversion factors, built only from `ase.units` (same rule as
# tests/python/test_ase_units.py).
H3 = Hartree / Bohr**3  # Hartree/bohr^3 -> eV/Angstrom^3
H4 = Hartree / Bohr**4  # Hartree/bohr^4 -> eV/Angstrom^4
RTOL_UNITS = 1.0e-12  # float round-off from one multiply, never a physical tolerance


# ---------------------------------------------------------------------------
# Fixtures: geometries and calculators.
#
# Non-equilibrium geometries throughout -- at a stationary point many
# third/fourth-derivative channels cancel by symmetry and a missing term would
# not show up.
# ---------------------------------------------------------------------------
WATER = ([8, 1, 1], [[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [-0.24, 0.92, 0.0]])
NONEQ_WATER = ([8, 1, 1], [[0.0, 0.0, 0.0], [1.05, 0.55, 0.0], [-0.60, 0.95, 0.10]])
SKEW_HF = ([9, 1], [[0.0, 0.0, 0.0], [0.70, 0.54, 0.40]])
#: Non-equilibrium formaldehyde: at 10000 K, 10 of its 12 orbitals are
#: fractionally occupied (1.99999 ... 1.79 ... 0.23 ... 0.0002) with no
#: degeneracy -- the cheap heavily smeared fixture of ``tests/smearing.rs``.
NONEQ_HCHO = (
    [6, 8, 1, 1],
    [[0.0, 0.0, 0.0], [1.28, 0.10, 0.05], [-0.60, 0.95, 0.10], [-0.62, -0.90, 0.12]],
)
#: 2-atom primitive fcc cell of diamond, a = 3.567 A -- the Grueneisen fixture of
#: ``tests/pbc_third_derivative.rs``.
DIAMOND_A = 3.567


def skew_direction(ndof: int) -> np.ndarray:
    """The skew probe direction of the Rust gates: no zero components, no
    accidental symmetry with any Cartesian axis."""

    return np.array(
        [0.23 - 0.11 * (k % 4) + 0.05 * ((k * 7) % 5) for k in range(ndof)], dtype=float
    )


def tight(**kwargs):
    """A native calculator with the SCF converged far past finite-difference noise."""

    settings = dict(energy_tolerance=1.0e-12, charge_tolerance=1.0e-10, max_scc=500)
    settings.update(kwargs)
    return gfn1_rs.Gfn1NativeCalculator(**settings)


@pytest.fixture(scope="module")
def calc():
    """T = 0, tight SCF: the reference state of the closed-form derivative ladder."""

    return tight(electronic_temperature=0.0)


@pytest.fixture(scope="module")
def smeared():
    """Heavily Fermi-smeared, dispersion off -- the option set of the Rust
    smearing gates (their finite-difference references pair with it term for
    term)."""

    return tight(
        electronic_temperature=10000.0,
        enable_dispersion=False,
        energy_tolerance=1.0e-14,
        charge_tolerance=1.0e-12,
    )


def diamond_atoms() -> Atoms:
    a = DIAMOND_A / 2.0
    return Atoms(
        "C2",
        positions=[[0.0, 0.0, 0.0], [DIAMOND_A / 4.0] * 3],
        cell=[[0.0, a, a], [a, 0.0, a], [a, a, 0.0]],
        pbc=True,
    )


#: Real-space cutoffs (bohr) reduced from the library defaults (30 / 40 / 10) for
#: speed. The Rust gates document the effect on the diamond mode Grueneisen
#: parameter: 0.90542 at the defaults, 0.90562 at the leanest cutoffs tried -- a
#: 0.02% spread, far inside every window asserted here.
PBC_CUTOFFS = dict(ao_cutoff=16.0, ewald_real_cutoff=24.0, ewald_sr_cutoff=10.0)


# ---------------------------------------------------------------------------
# FC3: Dense vs Vector vs Block, closed form and semi-numerical.
# ---------------------------------------------------------------------------
class TestThirdDerivative:
    def test_dense_contracted_equals_vector_mode(self, calc):
        """The Dense tensor contracted along ``v`` must reproduce the Vector mode.

        Not a tautology: Dense sums the un-symmetrised closed-form slabs into the
        packed store and 6-permutation-averages them, while Vector builds
        ``(A + B + B^T)/3`` from those same slabs without ever forming the
        ``3N^3`` tensor. Agreement pins the two assemblies against each other.
        """

        numbers, positions = WATER
        v = skew_direction(3 * len(numbers))
        dense = np.asarray(
            calc.third_derivative(numbers=numbers, positions=positions, unit="angstrom"),
            dtype=float,
        )
        vector = np.asarray(
            calc.third_derivative_vector(
                numbers=numbers, positions=positions, direction=v.tolist(), unit="angstrom"
            ),
            dtype=float,
        )
        assert dense.shape == (9, 9, 9)
        # Documented packing: dense[c][a][b] = T_abc.
        contracted = np.einsum("c,cab->ab", v, dense)
        scale = np.abs(vector).max()
        assert scale > 1.0e-3, f"cubic force constants look degenerate (scale {scale:.3e})"
        np.testing.assert_allclose(contracted, vector, rtol=0.0, atol=1.0e-9 * scale)

    def test_seminumerical_dense_matches_the_closed_form(self, calc):
        """The new semi-numerical Dense mode (central FD of the analytic Hessian
        along every DOF) against the strict closed form -- they are independent
        assemblies, so this is a real cross-check, loose only by the finite-
        difference truncation."""

        numbers, positions = WATER
        closed = np.asarray(
            calc.third_derivative(numbers=numbers, positions=positions, unit="angstrom"),
            dtype=float,
        )
        semi = np.asarray(
            calc.third_derivative_seminumerical(
                numbers=numbers, positions=positions, unit="angstrom", step=1.0e-3
            ),
            dtype=float,
        )
        assert semi.shape == closed.shape
        scale = np.abs(closed).max()
        worst = np.abs(semi - closed).max()
        assert worst < 1.0e-3 * scale, (
            f"semi-numerical vs closed-form FC3: worst {worst:.3e} (scale {scale:.3e})"
        )

    def test_seminumerical_block_is_exactly_the_dense_subblock(self, calc):
        """Block mode uses the same canonical packing as Dense (each unordered
        triple differenced along its largest index), so it must be bit-for-bit
        the sub-block -- not merely equal to FD truncation order."""

        numbers, positions = WATER
        dense = np.asarray(
            calc.third_derivative_seminumerical(
                numbers=numbers, positions=positions, unit="angstrom", step=1.0e-3
            ),
            dtype=float,
        )
        dofs, block = calc.third_derivative_seminumerical_block(
            numbers=numbers, positions=positions, atoms=[0, 2], unit="angstrom", step=1.0e-3
        )
        assert list(dofs) == [0, 1, 2, 6, 7, 8]
        block = np.asarray(block, dtype=float)
        idx = np.asarray(dofs)
        expected = dense[np.ix_(idx, idx, idx)]
        np.testing.assert_array_equal(block, expected)


# ---------------------------------------------------------------------------
# Finite-temperature FC3.
# ---------------------------------------------------------------------------
class TestThirdDerivativeFiniteT:
    def test_directional_equals_the_zero_temperature_assembly(self, calc):
        """At T = 0 the occupation-agnostic route must equal the adjoint-assembled
        Vector mode contracted ``vvv`` -- the equality gate that pins the
        finite-temperature assembly's term inventory."""

        numbers, positions = WATER
        v = skew_direction(3 * len(numbers))
        directional = calc.third_derivative_finite_t_directional(
            numbers=numbers, positions=positions, direction=v.tolist(), unit="angstrom"
        )
        k = np.asarray(
            calc.third_derivative_vector(
                numbers=numbers, positions=positions, direction=v.tolist(), unit="angstrom"
            ),
            dtype=float,
        )
        reference = float(v @ k @ v)
        assert abs(reference) > 1.0e-3
        assert directional == pytest.approx(reference, rel=1.0e-6)

    def test_dense_reconstructs_the_directional_value(self, calc):
        """The Dense finite-temperature tensor is recovered from directional
        evaluations by the cubic polarization identity; contracting it back must
        return the directional value (a 7-term alternating sum per element, so
        this is a relative check, not machine precision). Run on a diatomic to
        keep the ~C(n+2, 3) directional evaluations cheap."""

        numbers, positions = SKEW_HF
        ndof = 3 * len(numbers)
        v = skew_direction(ndof)
        dense = np.asarray(
            calc.third_derivative_finite_t(
                numbers=numbers, positions=positions, unit="angstrom"
            ),
            dtype=float,
        )
        assert dense.shape == (ndof, ndof, ndof)
        contracted = float(np.einsum("cab,a,b,c->", dense, v, v, v))
        directional = calc.third_derivative_finite_t_directional(
            numbers=numbers, positions=positions, direction=v.tolist(), unit="angstrom"
        )
        assert abs(directional) > 1.0e-3
        assert contracted == pytest.approx(directional, rel=1.0e-7)

    def test_block_matches_the_dense_subblock(self, calc):
        numbers, positions = SKEW_HF
        dense = np.asarray(
            calc.third_derivative_finite_t(
                numbers=numbers, positions=positions, unit="angstrom"
            ),
            dtype=float,
        )
        dofs = [0, 2, 4]
        out_dofs, block = calc.third_derivative_finite_t_block(
            numbers=numbers, positions=positions, dofs=dofs, unit="angstrom"
        )
        assert list(out_dofs) == dofs
        idx = np.asarray(dofs)
        np.testing.assert_allclose(
            np.asarray(block, dtype=float), dense[np.ix_(idx, idx, idx)], rtol=1.0e-9, atol=0.0
        )

    def test_smeared_fixture_matches_the_hessian_finite_difference(self, smeared, calc):
        """The real finite-temperature gate: on non-equilibrium formaldehyde at
        10000 K (10 of 12 orbitals fractionally occupied) the analytic directional
        third derivative must equal the central FD of the **smeared** analytic
        Hessian contracted ``vv``.

        The finite difference is taken in **bohr** (positions handed over as
        ``unit="bohr"``), because that is the unit the returned Hartree/bohr^3
        value differentiates in. A T = 0 run on the same geometry is included to
        show the smearing really moves the answer, i.e. that this is not a
        vacuous re-derivation of the integer-occupation path.
        """

        numbers, positions_ang = NONEQ_HCHO
        ndof = 3 * len(numbers)
        v = skew_direction(ndof)
        positions = (np.asarray(positions_ang, dtype=float) * gfn1_rs.ANGSTROM_TO_BOHR)

        analytic = smeared.third_derivative_finite_t_directional(
            numbers=numbers, positions=positions.tolist(), direction=v.tolist(), unit="bohr"
        )
        assert np.isfinite(analytic) and abs(analytic) > 1.0e-3

        def hessian_vv(shift: float) -> float:
            displaced = positions + shift * v.reshape(-1, 3)
            h = np.asarray(
                smeared.hessian(numbers=numbers, positions=displaced.tolist(), unit="bohr"),
                dtype=float,
            )
            return float(v @ h @ v)

        step = 1.0e-3
        fd = (hessian_vv(step) - hessian_vv(-step)) / (2.0 * step)
        assert analytic == pytest.approx(fd, rel=2.0e-5), (
            f"smeared directional FC3 {analytic:.10e} vs Hessian FD {fd:.10e}"
        )

        cold = calc.third_derivative_finite_t_directional(
            numbers=numbers, positions=positions.tolist(), direction=v.tolist(), unit="bohr"
        )
        assert abs(cold - analytic) > 1.0e-3 * abs(analytic), (
            "the smeared and T = 0 cubic constants are indistinguishable -- "
            "electronic_temperature is not reaching the assembly"
        )


# ---------------------------------------------------------------------------
# FC4.
# ---------------------------------------------------------------------------
class TestFourthDerivative:
    def test_analytic_directional_matches_the_seminumerical_reference(self, calc):
        """The analytic quartic against its verification reference -- the central
        FD of the analytic *third* derivative along the same direction, with
        everything reconverged at ``R +/- h v``. One step, loose window: the Rust
        gate additionally checks the ``h^2`` scaling of the residual."""

        numbers, positions = NONEQ_WATER
        v = skew_direction(3 * len(numbers))
        analytic = calc.fourth_derivative_directional(
            numbers=numbers, positions=positions, direction=v.tolist(), unit="angstrom"
        )
        semi = calc.fourth_derivative_directional_seminumerical(
            numbers=numbers,
            positions=positions,
            direction=v.tolist(),
            unit="angstrom",
            step=1.0e-3,
        )
        assert np.isfinite(analytic) and abs(analytic) > 1.0e-3
        assert abs(analytic - semi) < 1.0e-5 * (1.0 + abs(semi)), (
            f"analytic quartic {analytic:.10e} vs seminumerical {semi:.10e}"
        )

    def test_block_reconstructs_the_directional_quartic(self, calc):
        """A direction supported only on the block's DOFs contracts the block
        tensor to exactly the directional quartic for that direction -- the cheap
        way to gate the mixed-index polarization reconstruction."""

        numbers, positions = NONEQ_WATER
        ndof = 3 * len(numbers)
        dofs = [0, 1, 4]
        out_dofs, block = calc.fourth_derivative_block(
            numbers=numbers, positions=positions, dofs=dofs, unit="angstrom"
        )
        assert list(out_dofs) == dofs
        block = np.asarray(block, dtype=float)
        assert block.shape == (len(dofs), len(dofs), len(dofs), len(dofs))

        w = np.array([0.31, -0.17, 0.44])
        v = np.zeros(ndof)
        v[dofs] = w
        contracted = float(np.einsum("abcd,a,b,c,d->", block, w, w, w, w))
        directional = calc.fourth_derivative_directional(
            numbers=numbers, positions=positions, direction=v.tolist(), unit="angstrom"
        )
        assert abs(directional) > 1.0e-4
        assert contracted == pytest.approx(directional, rel=1.0e-7)

    def test_expanded_tensor_size_cap_is_enforced(self, calc):
        """The ``n^4`` expansion is capped at ``MAX_FOURTH_DERIVATIVE_NDOF``; the
        guard fires before the SCF, so this test is cheap."""

        cap = gfn1_rs.MAX_FOURTH_DERIVATIVE_NDOF
        nat = cap // 3 + 2
        numbers = [1] * nat
        positions = [[1.2 * i, 0.0, 0.0] for i in range(nat)]
        with pytest.raises(ValueError, match="capped at"):
            calc.fourth_derivative(numbers=numbers, positions=positions, unit="angstrom")
        # ...but a small BLOCK of the same oversized system is legal (the cap is
        # applied to |dofs|, not to 3N). Only the guard is exercised here.
        assert cap >= 3


# ---------------------------------------------------------------------------
# Periodic third derivative and Grueneisen parameters.
# ---------------------------------------------------------------------------
class TestPeriodic:
    def test_gruneisen_of_diamond(self):
        """Diamond's optical mode Grueneisen parameter. The Rust gate records
        0.90542 at the library cutoffs and 0.90562 at the leanest ones; experiment
        for diamond is 0.9 - 1.2, so GFN1 lands at the bottom of the literature
        range. Window +/- 0.05 -- a sanity gate, not a fit."""

        atoms = diamond_atoms()
        atoms.calc = GFN1RSCalculator(
            electronic_temperature=0.0,
            energy_tolerance=1.0e-11,
            charge_tolerance=1.0e-10,
            max_scc=500,
        )
        # `atoms.calc = ...` does not populate `calc.atoms` until a calculation
        # has run, so hand the Atoms over explicitly.
        out = atoms.calc.get_gruneisen(
            atoms=atoms, delta=5.0e-3, temperatures=(300.0,), second_order=True, **PBC_CUTOFFS
        )

        mode_gamma = out["mode_gamma"]
        assert mode_gamma.shape == (6,)
        # The three acoustic branches are excluded and carry NaN by construction.
        assert np.all(np.isnan(mode_gamma[: out["acoustic_modes"]]))
        optical = mode_gamma[out["acoustic_modes"] :]
        assert np.all(np.isfinite(optical))
        assert optical.mean() == pytest.approx(0.905, abs=0.05), (
            f"diamond mode Grueneisen parameters {optical}"
        )
        # The triply degenerate optical mode: one shared subspace-averaged value.
        np.testing.assert_allclose(optical, optical[0], rtol=1.0e-3)
        # Mode assignment must be clean (no crossing under this small a strain).
        assert out["min_optical_overlap"] > 0.99

        gamma_th = out["thermodynamic_gamma"]
        assert gamma_th.shape == (1, 2)
        assert gamma_th[0, 0] == pytest.approx(300.0)
        assert gamma_th[0, 1] == pytest.approx(0.905, abs=0.05)

        # second_order=True: the refit gamma reproduces the two-point one, and the
        # curvature/thermodynamic tables are populated rather than NaN/empty.
        assert out["second_order_stencil"] == "three_point"
        refit = out["mode_gamma_refit"][out["acoustic_modes"] :]
        np.testing.assert_allclose(refit, optical, rtol=1.0e-4)
        assert np.all(np.isfinite(out["mode_gamma2"][out["acoustic_modes"] :]))
        assert out["thermodynamic_gamma2"].shape == (1, 2)
        assert out["thermodynamic_gamma2_full"].shape == (1, 2)

        # Frequencies stay in cm^-1 and the volume is converted to Angstrom^3.
        assert out["frequencies_cm1"][-1] > 1000.0
        assert out["volume"] == pytest.approx(out["volume_bohr3"] * Bohr**3, rel=RTOL_UNITS)
        # ...and that really is the ASE cell volume. Loose only because the native
        # Angstrom->bohr constant and `ase.units.Bohr` are different CODATA sets
        # (~4e-10 relative each way, cubed here).
        assert out["volume"] == pytest.approx(diamond_atoms().get_volume(), rel=1.0e-7)
        # Grueneisen parameters are dimensionless: strain-expanded frequencies must
        # be LOWER than the reference ones for a positive gamma.
        assert np.all(
            out["frequencies_cm1_expanded"][out["acoustic_modes"] :]
            < out["frequencies_cm1"][out["acoustic_modes"] :]
        )

    def test_periodic_third_derivative_vector_is_an_exact_contraction(self):
        """Vector mode accumulates the same per-DOF central differences the Dense
        mode builds, in the same index order, so it must be bit-for-bit the
        contraction -- not merely equal to FD truncation order (which is what a
        single displacement *along* ``v`` would give)."""

        atoms = diamond_atoms()
        native = tight(electronic_temperature=0.0, energy_tolerance=1.0e-11)
        numbers = atoms.get_atomic_numbers().astype(np.uint8).tolist()
        positions = atoms.get_positions().tolist()
        cell = np.asarray(atoms.get_cell(), dtype=float).tolist()

        slabs = np.asarray(
            native.third_derivative_periodic(
                numbers=numbers,
                positions=positions,
                cell=cell,
                unit="angstrom",
                step=1.0e-3,
                **PBC_CUTOFFS,
            ),
            dtype=float,
        )
        assert slabs.shape == (6, 6, 6)
        scale = np.abs(slabs).max()
        assert scale > 0.1 and np.isfinite(scale)

        # Translating every atom along one Cartesian axis maps the crystal onto
        # itself, so the slabs of that axis must sum to zero: a genuine
        # cancellation across 12 independently converged SCC + Hessian runs.
        for axis in range(3):
            residual = np.abs(slabs[axis::3].sum(axis=0)).max()
            assert residual < 1.0e-5, f"acoustic sum rule residual {residual:.3e} (axis {axis})"

        # A deliberately sparse direction also exercises the zero-component skip.
        v = np.zeros(6)
        v[1] = 0.7
        v[4] = -1.3
        k = np.asarray(
            native.third_derivative_periodic_vector(
                numbers=numbers,
                positions=positions,
                cell=cell,
                direction=v.tolist(),
                unit="angstrom",
                step=1.0e-3,
                **PBC_CUTOFFS,
            ),
            dtype=float,
        )
        np.testing.assert_allclose(k, np.einsum("c,cab->ab", v, slabs), rtol=0.0, atol=1.0e-12)


# ---------------------------------------------------------------------------
# ASE unit contract of the three new calculator methods. Every assertion is of
# the form `ase_value == native_value * exact_ase_units_factor`, so it fails if a
# conversion is dropped, doubled, or taken from a different CODATA set.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pair():
    """A non-equilibrium water `Atoms` on the ASE calculator, paired with an
    identically configured native calculator to convert against."""

    numbers, positions = NONEQ_WATER
    atoms = Atoms(numbers=numbers, positions=positions)
    atoms.calc = GFN1RSCalculator(
        electronic_temperature=0.0,
        energy_tolerance=1.0e-12,
        charge_tolerance=1.0e-10,
        max_scc=500,
    )
    return atoms, tight(electronic_temperature=0.0)


class TestAseUnits:
    def test_module_constant_comes_from_ase_units(self):
        from gfn1_rs import ase as ase_mod

        assert ase_mod._FOURTH == Hartree / Bohr**4

    def test_finite_t_third_directional_is_ev_per_angstrom_cubed(self, pair):
        atoms, native = pair
        numbers, positions = NONEQ_WATER
        v = skew_direction(3 * len(numbers))
        raw = native.third_derivative_finite_t_directional(
            numbers=numbers, positions=positions, direction=v.tolist(), unit="angstrom"
        )
        got = atoms.calc.get_third_derivative_finite_t(v, atoms)
        assert got == pytest.approx(raw * H3, rel=RTOL_UNITS)

    def test_finite_t_third_block_is_ev_per_angstrom_cubed(self, pair):
        atoms, native = pair
        numbers, positions = NONEQ_WATER
        dofs = [0, 3]
        _, raw = native.third_derivative_finite_t_block(
            numbers=numbers, positions=positions, dofs=dofs, unit="angstrom"
        )
        out_dofs, got = atoms.calc.get_third_derivative_finite_t(atoms=atoms, dofs=dofs)
        assert out_dofs == dofs
        np.testing.assert_allclose(
            got, np.asarray(raw, dtype=float) * H3, rtol=RTOL_UNITS, atol=0.0
        )

    def test_fourth_derivative_directional_is_ev_per_angstrom_fourth(self, pair):
        atoms, native = pair
        numbers, positions = NONEQ_WATER
        v = skew_direction(3 * len(numbers))
        raw = native.fourth_derivative_directional(
            numbers=numbers, positions=positions, direction=v.tolist(), unit="angstrom"
        )
        got = atoms.calc.get_fourth_derivative_directional(v, atoms)
        assert got == pytest.approx(raw * H4, rel=RTOL_UNITS)

        raw_semi = native.fourth_derivative_directional_seminumerical(
            numbers=numbers,
            positions=positions,
            direction=v.tolist(),
            unit="angstrom",
            step=1.0e-3,
        )
        got_semi = atoms.calc.get_fourth_derivative_directional(
            v, atoms, method="seminumerical", step=1.0e-3
        )
        assert got_semi == pytest.approx(raw_semi * H4, rel=RTOL_UNITS)

    def test_unknown_fourth_derivative_method_is_rejected(self, pair):
        atoms, _ = pair
        with pytest.raises(ValueError, match="unknown fourth-derivative method"):
            atoms.calc.get_fourth_derivative_directional(
                np.zeros(9), atoms, method="magic"
            )

    def test_finite_t_third_rejects_direction_and_dofs_together(self, pair):
        atoms, _ = pair
        with pytest.raises(ValueError, match="not both"):
            atoms.calc.get_third_derivative_finite_t(np.zeros(9), atoms, dofs=[0])

    def test_gruneisen_requires_a_periodic_atoms(self, pair):
        atoms, _ = pair
        with pytest.raises(ValueError, match="periodic"):
            atoms.calc.get_gruneisen(atoms=atoms)
