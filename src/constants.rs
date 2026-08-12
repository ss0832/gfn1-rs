// SPDX-License-Identifier: GPL-3.0-or-later

/// Hartree energy in electronvolt: 27.211386245988 eV (CODATA 2018). Before
/// v0.5.0 this was the CODATA-2010 value 27.21138505.
///
/// This is the **unit-reporting** conversion: it is what turns computed
/// energies, excitation energies, forces and Hessians into eV / eV·Å⁻¹ /
/// eV·Å⁻² for output, so it should track the current CODATA recommendation.
///
/// It is deliberately **not** the constant used to read the GFN1 parameter
/// file. GFN1-xTB was parametrized against the older 1/27.21138505 factor, and
/// xtb/tblite (via mctc-lib) keep using it for that conversion; the model-side
/// constant therefore lives separately in [`crate::basis::EV_TO_HARTREE`] and
/// must stay on the legacy value. Switching the model side to CODATA 2018
/// measurably breaks tblite parity (~2.0e-6 Eh on caffeine, against a 1e-6
/// parity tolerance), so the two constants are distinct on purpose.
pub const HARTREE_TO_EV: f64 = 27.211_386_245_988;
/// Reciprocal of [`HARTREE_TO_EV`] (CODATA 2018). Unit reporting only — see the
/// note there on why [`crate::basis::EV_TO_HARTREE`] stays on the legacy value.
pub const EV_TO_HARTREE: f64 = 1.0 / HARTREE_TO_EV;
/// Boltzmann constant in Hartree/K: exact SI k_B = 1.380649e-23 J/K divided by
/// the CODATA-2018 Hartree energy 4.3597447222071e-18 J. One shared value for
/// every finite-temperature path (restricted SCC, spin-polarized SCC, CPXTB,
/// periodic SCC/Hessian) — before v0.5.0 the restricted and spin paths used two
/// slightly different constants, breaking their byte-identity at T > 0.
pub const KB_HARTREE_PER_K: f64 = 3.166_811_563_455_6e-6;
pub const ANGSTROM_TO_BOHR: f64 = 1.889_726_124_625_770_2;
pub const BOHR_TO_ANGSTROM: f64 = 1.0 / ANGSTROM_TO_BOHR;
pub const FORCE_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM: f64 = HARTREE_TO_EV / BOHR_TO_ANGSTROM;
pub const HESSIAN_HARTREE_PER_BOHR2_TO_EV_PER_ANGSTROM2: f64 =
    HARTREE_TO_EV / (BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM);

#[cfg(test)]
mod tests {
    #[test]
    fn kb_hartree_matches_si_exact_ratio() {
        // k_B = 1.380649e-23 J/K (exact, SI 2019); E_h = 4.3597447222071e-18 J (CODATA 2018).
        let kb_si = 1.380_649e-23_f64;
        let hartree_j = 4.359_744_722_207_1e-18_f64;
        assert!(
            (super::KB_HARTREE_PER_K - kb_si / hartree_j).abs() < 1.0e-16,
            "KB_HARTREE_PER_K drifted from the SI-exact ratio"
        );
    }
}

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
