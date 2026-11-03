"""Independent scalar-energy differences for public magnetic tensor components."""
from functools import lru_cache
import numpy as np
from gfn1_rs import Gfn1NativeCalculator

NUMBERS = [8, 1, 1]
POSITIONS = np.array([[.12, -.09, .03], [1.43, .91, .34], [-1.21, 1.08, -.22]])
SETTINGS = dict(electronic_temperature=0., energy_tolerance=1e-13,
                charge_tolerance=1e-11, max_scc=1000)


def test_magnetic_gradient_and_all_tensor_components_from_energy():
    calc = Gfn1NativeCalculator(**SETTINGS)
    b = (.017, -.023, .031)
    result = calc.magnetic_forces(NUMBERS, POSITIONS.tolist(), b, unit="bohr", step=2e-4)
    def energy_pos(displacement, coordinate):
        pos = POSITIONS.copy()
        pos.ravel()[coordinate] += displacement
        return calc.magnetic_energy(NUMBERS, pos.tolist(), b, unit="bohr")
    errors = []
    for coordinate in range(9):
        h = 8e-4
        coarse = (energy_pos(h, coordinate)-energy_pos(-h, coordinate))/(2*h)
        fine = (energy_pos(h/2, coordinate)-energy_pos(-h/2, coordinate))/h
        errors.append(abs(np.asarray(result["gradient"]).ravel()[coordinate]-(4*fine-coarse)/3))
    assert max(errors) < 1e-7
    np.testing.assert_allclose(result["forces"], -np.asarray(result["gradient"]), atol=0)
    def energy(b):
        return calc.magnetic_energy(NUMBERS, POSITIONS.tolist(), tuple(b), unit="bohr")
    h = .003
    e0 = energy(np.zeros(3))
    xi_fd = np.zeros((3, 3))
    for i in range(3):
        for j in range(i+1):
            def derivative(h):
                u = np.eye(3)[i]*h
                v = np.eye(3)[j]*h
                if i == j:
                    return -(energy(u)-2*e0+energy(-u))/h**2
                return -(energy(u+v)-energy(u-v)-energy(-u+v)+energy(-u-v))/(4*h*h)
            xi_fd[i,j] = xi_fd[j,i] = (4*derivative(h/2)-derivative(h))/3
    factor = 78.910366008
    xi = np.asarray(calc.magnetizability_tensor(NUMBERS, POSITIONS.tolist(), unit="bohr"))/factor
    np.testing.assert_allclose(xi, xi_fd, atol=3e-6, rtol=2e-5)
    diagonal = np.asarray(calc.magnetizability_diagonal(NUMBERS, POSITIONS.tolist(), unit="bohr"))/factor
    np.testing.assert_allclose(diagonal, np.diag(xi), atol=1e-12)
    np.testing.assert_allclose(calc.magnetizability(NUMBERS, POSITIONS.tolist(), unit="bohr")/factor,
                               np.trace(xi)/3, atol=1e-12)
    l = np.asarray(calc.angular_momentum(NUMBERS, POSITIONS.tolist(), unit="bohr"))
    np.testing.assert_allclose(l+l.transpose(0,2,1), 0, atol=1e-12)
    dipole = calc.lao_dipole(NUMBERS, POSITIONS.tolist(), b_field=b, unit="bohr")
    d = np.asarray(dipole["re"])+1j*np.asarray(dipole["im"])
    np.testing.assert_allclose(d, d.conj().transpose(0,2,1), atol=1e-12)
    assert np.isfinite(calc.nmr_shielding(NUMBERS, POSITIONS.tolist(), nucleus=0, unit="bohr")).all()
    print("magnetic all 9 nuclear/max xi errors", max(errors), np.max(abs(xi-xi_fd)))


def test_all_cotton_mouton_components_from_fourth_energy_differences():
    calc = Gfn1NativeCalculator(**SETTINGS)
    @lru_cache(None)
    def field_calc(e):
        return Gfn1NativeCalculator(**SETTINGS, electric_field=e)
    @lru_cache(None)
    def energy(b, e):
        return field_calc(e).magnetic_energy(NUMBERS, POSITIONS.tolist(), b, unit="bohr")
    def alpha(b, e_step):
        a = np.zeros((3,3))
        zero = (0.,0.,0.)
        for i in range(3):
            for j in range(i+1):
                u = np.eye(3)[i]*e_step
                v = np.eye(3)[j]*e_step
                if i == j:
                    value = -(energy(b,tuple(u))-2*energy(b,zero)+energy(b,tuple(-u)))/e_step**2
                else:
                    value = -(energy(b,tuple(u+v))-energy(b,tuple(u-v))-energy(b,tuple(-u+v))+energy(b,tuple(-u-v)))/(4*e_step**2)
                a[i,j] = a[j,i] = value
        return a
    def cm_energy(e_step,b_step):
        zero = (0.,0.,0.)
        a0 = alpha(zero,e_step)
        return np.array([(alpha(tuple(u),e_step)-2*a0+alpha(tuple(-u),e_step))/b_step**2 for u in np.eye(3)*b_step])
    coarse = cm_energy(.004,.04)
    fine = cm_energy(.002,.02)
    reference = (4*fine-coarse)/3
    api_coarse = np.asarray(calc.cotton_mouton(NUMBERS, POSITIONS.tolist(), unit="bohr", e_step=.002,b_step=.04))
    api_fine = np.asarray(calc.cotton_mouton(NUMBERS, POSITIONS.tolist(), unit="bohr", e_step=.001,b_step=.02))
    actual = (4*api_fine-api_coarse)/3
    np.testing.assert_allclose(actual, reference, atol=3e-4, rtol=2e-3)
    np.testing.assert_allclose(calc.mcd(NUMBERS, POSITIONS.tolist(), unit="bohr"), 0, atol=1e-10)
    np.testing.assert_allclose(calc.magnetic_polarizability(NUMBERS, POSITIONS.tolist(), unit="bohr", e_step=.001),
                               alpha((0.,0.,0.),.002),atol=2e-4,rtol=2e-4)
    print("Cotton-Mouton all 27 components max error",np.max(abs(actual-reference)))
