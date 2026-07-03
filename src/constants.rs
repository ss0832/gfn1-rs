// SPDX-License-Identifier: GPL-3.0-or-later

pub const EV_TO_HARTREE: f64 = 1.0 / 27.211_385_05;
pub const HARTREE_TO_EV: f64 = 27.211_385_05;
pub const ANGSTROM_TO_BOHR: f64 = 1.889_726_124_625_770_2;
pub const BOHR_TO_ANGSTROM: f64 = 1.0 / ANGSTROM_TO_BOHR;
pub const FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM: f64 = HARTREE_TO_EV / BOHR_TO_ANGSTROM;
pub const HESSIAN_HARTREE_PER_BOHR2_TO_EV_PER_ANGSTROM2: f64 =
    HARTREE_TO_EV / (BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM);

pub fn covalent_radius_bohr(z: u8) -> f64 {
    // Cordero/Pyykko-like covalent radii in Angstrom for CN damping. These are
    // generic element data, not xTB model parameters.
    const RAD_A: [f64; 87] = [
        0.0, 0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58, 1.66, 1.41, 1.21, 1.11,
        1.07, 1.05, 1.02, 1.06, 2.03, 1.76, 1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32,
        1.22, 1.22, 1.20, 1.19, 1.20, 1.20, 1.16, 2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46,
        1.42, 1.39, 1.45, 1.44, 1.42, 1.39, 1.39, 1.38, 1.39, 1.40, 2.44, 2.15, 2.07, 2.04, 2.03,
        2.01, 1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, 1.87, 1.75, 1.70, 1.62,
        1.51, 1.44, 1.41, 1.36, 1.36, 1.32, 1.45, 1.46, 1.48, 1.40, 1.50, 1.50,
    ];
    RAD_A.get(z as usize).copied().unwrap_or(1.5) * ANGSTROM_TO_BOHR
}
