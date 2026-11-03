"""The public spectrum API must work without official executables or PATH."""
import inspect
import numpy as np
from gfn1_rs import Gfn1NativeCalculator



def test_tda_accepts_fractional_filling_and_matches_finite_difference():
    calc = Gfn1NativeCalculator(electronic_temperature=30000., max_scc=1000,
                               energy_tolerance=1e-13, charge_tolerance=1e-11)
    numbers, positions = [1, 1], [[0, 0, 0], [.74, 0, 0]]
    calc.tda(numbers, positions)
    analytic = calc.tda_gradient(numbers, positions, method="analytic")
    numerical = calc.tda_gradient(numbers, positions, method="fd", step=1e-4)
    np.testing.assert_allclose(analytic["gradient"], numerical["gradient"], atol=1e-7)


def test_stda_is_native_without_external_programs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("XTB4STDAHOME", raising=False)
    calc = Gfn1NativeCalculator(charge=0)
    signature = inspect.signature(calc.stda_xtb)
    assert not hasattr(calc, "stda_xtb_reference")
    assert "xtb4stda_executable" not in signature.parameters
    assert "std2_executable" not in signature.parameters
    result = calc.stda_xtb(
        [8, 1, 1], [[0, 0, 0], [.757, .586, 0], [-.757, .586, 0]],
        n_states=10, energy_window_ev=20.0, triplet=False,
    )
    assert result["implementation"] == "native-rust"
    assert result["model"] == "sTDA-xTB"
    assert result["electronic_temperature_kelvin"] == 300.
    assert len(result["excitation_energies_ev"]) == 10
    np.testing.assert_allclose(result["excitation_energies_ev"][:3],
                               [7.5134308877, 8.9575029853, 9.6060338330], atol=2e-6)
    assert list(tmp_path.iterdir()) == []


def test_stda_uses_requested_temperature_instead_of_ignoring_it():
    import pytest
    nums = [8, 1, 1]
    pos = [[0, 0, 0], [.757, .586, 0], [-.757, .586, 0]]
    for temperature in (0., 300.):
        result = Gfn1NativeCalculator(electronic_temperature=temperature).stda_xtb(nums,pos)
        assert result["electronic_temperature_kelvin"] == temperature
        assert result["maximum_occupation_deviation"] < 1e-10
    hot = Gfn1NativeCalculator(electronic_temperature=100000.).stda_xtb(nums,pos,energy_window_ev=20.)
    assert hot["electronic_temperature_kelvin"] == 100000.
    assert hot["maximum_occupation_deviation"] > 1e-3
    assert hot["occupation_model"] == "fractional canonical ensemble interband CI"
    assert abs(hot["ci_eigenvalues_hartree"][0]-result["ci_eigenvalues_hartree"][0]) > 1e-5
    with pytest.raises(ValueError,match="before integral allocation"):
        Gfn1NativeCalculator().stda_xtb(nums,pos,memory_budget_bytes=1,max_ao=1000000)


def test_native_stda_total_and_excitation_gradient_and_ase_units():
    from ase import Atoms
    from ase.units import Hartree, Bohr
    from gfn1_rs import GFN1RSCalculator
    calc = Gfn1NativeCalculator(electronic_temperature=300., energy_tolerance=1e-13,
                               charge_tolerance=1e-11, max_scc=1000)
    numbers = [8, 1, 1]
    positions = np.array([[.03, -.02, .01], [.757, .586, .02], [-.757, .586, -.01]])
    result = calc.stda_xtb_gradient(numbers, positions.tolist())
    assert result["implementation"] == "native-rust" and result["method"] == "analytic"
    assert result["model"] == "GFN1+sTDA-xTB"
    np.testing.assert_allclose(result["forces"], -np.asarray(result["gradient"]), atol=0)
    # Whole-computation finite differences are an independent validation only.
    from gfn1_rs import ANGSTROM_TO_BOHR
    positions_bohr = positions * ANGSTROM_TO_BOHR
    def energy(shift, coordinate):
        pos = positions_bohr.copy()
        pos.ravel()[coordinate] += shift
        return (calc.calculate(numbers, pos.tolist(), unit="bohr").energy_hartree
                + calc.stda_xtb(numbers, pos.tolist(), unit="bohr")["excitation_energies_hartree"][0])
    for coordinate in range(9):
        h = 1e-3
        coarse = (energy(h, coordinate) - energy(-h, coordinate)) / (2*h)
        fine = (energy(h/2, coordinate) - energy(-h/2, coordinate)) / h
        assert abs(np.asarray(result["gradient"]).ravel()[coordinate] - (4*fine-coarse)/3) < 1e-8
    atoms = Atoms(numbers=numbers, positions=positions)
    atoms.calc = GFN1RSCalculator(electronic_temperature=300., energy_tolerance=1e-13,
                                 charge_tolerance=1e-11, max_scc=1000)
    ase = atoms.calc.get_stda_xtb_gradient(atoms=atoms)
    np.testing.assert_allclose(ase["gradient"], np.asarray(result["gradient"])*Hartree/Bohr, atol=1e-11)
    np.testing.assert_allclose(ase["excitation_gradient"], np.asarray(result["excitation_gradient"])*Hartree/Bohr, atol=1e-11)


def test_tda_legacy_default_and_ao_option_match_ase():
    from ase import Atoms
    from gfn1_rs import GFN1RSCalculator
    calc = Gfn1NativeCalculator(electronic_temperature=0.)
    numbers = [8, 1, 1]
    positions = [[0, 0, 0], [.757, .586, 0], [-.757, .586, 0]]
    old = calc.tda(numbers, positions)
    explicit = calc.tda(numbers, positions, transition_dipole="mulliken")
    ao = calc.tda(numbers, positions, transition_dipole="ao_length")
    assert old == explicit
    assert ao["excitation_energies_hartree"] == old["excitation_energies_hartree"]
    assert old["oscillator_strengths"][0] < 1e-12 and ao["oscillator_strengths"][0] > 1e-5
    atoms = Atoms(numbers=numbers, positions=positions)
    atoms.calc = GFN1RSCalculator(electronic_temperature=0.)
    assert atoms.calc.get_tda(atoms=atoms, transition_dipole="ao_length") == ao


def test_degenerate_tda_gradient_and_hessian_are_explicit():
    import pytest

    numbers = [6, 1, 1, 1, 1]
    positions = [
        [0, 0, 0], [.6276, .6276, .6276], [.6276, -.6276, -.6276],
        [-.6276, .6276, -.6276], [-.6276, -.6276, .6276],
    ]
    calc = Gfn1NativeCalculator(electronic_temperature=0., threads=2)
    for method in ("analytic", "semi-numerical", "numerical"):
        with pytest.raises(ValueError, match="3-fold degenerate"):
            calc.tda_gradient(numbers, positions, method=method)
    with pytest.raises(ValueError, match="3-fold degenerate"):
        calc.tda_hessian(numbers, positions)
    gradient = calc.tda_gradient(numbers, positions, manifold_average=True)
    hessian = calc.tda_hessian(numbers, positions, manifold_average=True)
    assert gradient["manifold_size"] == hessian["manifold_size"] == 3
    assert gradient["manifold_status"] == hessian["manifold_status"] == "degenerate_mean"
    isolated = calc.tda_hessian([8, 1, 1], [[0, 0, 0], [.757, .586, 0], [-.757, .586, 0]],
                                manifold_average=True)
    assert isolated["manifold_size"] == 1
    assert isolated["manifold_status"] == "isolated_root"


def test_ase_passes_manifold_gradient_and_hessian_options():
    import pytest

    ase = pytest.importorskip("ase")
    from gfn1_rs import GFN1RSCalculator

    atoms = ase.Atoms("OH2", positions=[[0, 0, 0], [.757, .586, 0], [-.757, .586, 0]])
    calc = GFN1RSCalculator(electronic_temperature=0., threads=2)
    for method in (calc.get_tda_gradient, calc.get_tda_hessian):
        result = method(atoms=atoms, manifold_average=True)
        assert result["manifold_size"] == 1
        assert result["manifold_status"] == "isolated_root"


def test_thermal_rpa_and_stda_derivatives_share_manifold_metadata():
    from ase import Atoms
    from gfn1_rs import GFN1RSCalculator
    atoms = Atoms(numbers=[8, 1, 1], positions=[[0., 0., 0.], [.757, .586, 0.], [-.757, .586, 0.]])
    atoms.calc = GFN1RSCalculator(electronic_temperature=30000.)
    for name, options in (("rpa", {}), ("stda_xtb", {"thermal_response": True})):
        spectrum = getattr(atoms.calc, f"get_{name}")(atoms=atoms, **options)
        assert spectrum["excitation_energies_hartree"]
        gradient = getattr(atoms.calc, f"get_{name}_gradient")(atoms=atoms, manifold_average=True, **options)
        hessian = getattr(atoms.calc, f"get_{name}_hessian")(atoms=atoms, manifold_average=True, **options)
        assert gradient["manifold_status"] == hessian["manifold_status"] == "isolated_root"
        assert gradient["manifold_size"] == hessian["manifold_size"] == 1
        assert np.isfinite(hessian["hessian"]).all()
        assert hessian["hessian"].shape == (9, 9)


def test_cartesian_derivative_disk_storage_round_trips_all_components(tmp_path):
    from itertools import combinations_with_replacement
    from ase import Atoms
    from ase.units import Bohr, Hartree
    from gfn1_rs import GFN1RSCalculator

    nums = [1, 1]
    pos = [[0., 0., 0.], [.67, .28, .09]]
    calc = Gfn1NativeCalculator()
    atoms = Atoms(numbers=nums, positions=pos)
    atoms.calc = GFN1RSCalculator()
    for order, name in ((3, "third_derivative"), (4, "fourth_derivative")):
        method = getattr(calc, name)
        memory = np.asarray(method(nums, pos, disk_storage=False))
        path = tmp_path / f"native-fc{order}.bin"
        metadata = method(nums, pos, disk_storage=True, output_path=path)
        assert metadata["n_dof"] == 6 and metadata["order"] == order
        assert path.read_bytes()[:8] == f"GFN1FC{order}\0".encode()
        values = np.memmap(path, dtype="<f8", mode="r", offset=metadata["offset_bytes"],
                           shape=(metadata["packed_entries"],))
        canonical = sorted(combinations_with_replacement(range(6), order), key=lambda x: x[::-1])
        np.testing.assert_allclose(values, [memory[x] for x in canonical], atol=1e-12)
        ase_method = getattr(atoms.calc, f"get_{name}")
        ase_output = ase_method(atoms=atoms, disk_storage=True, output_path=tmp_path / f"ase-fc{order}.bin")
        ase_memory = ase_method(atoms=atoms, disk_storage=False)
        np.testing.assert_allclose(ase_memory, memory * Hartree / Bohr**order, atol=1e-10)
        assert ase_output["units"] == f"Hartree/bohr^{order}"
        ase_values = np.memmap(ase_output["path"], dtype="<f8", mode="r", offset=24)
        np.testing.assert_array_equal(ase_values, values)
