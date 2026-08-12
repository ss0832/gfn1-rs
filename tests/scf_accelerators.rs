// SPDX-License-Identifier: GPL-3.0-or-later
//! All SCC accelerators (and level shifting) must converge to the same energy.

use gfn1_rs::{run_electronic, ElectronicOptions, Gfn1Parameters, PeriodicSystem, SccAccelerator};

fn load_params() -> Option<Gfn1Parameters> {
    Some(Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed"))
}

const WATER: &str = "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n";
const BROMOETHANOL: &str = "9\nbromoethanol\n\
     C 0.000000 0.000000 0.000000\n\
     C 1.520000 0.000000 0.000000\n\
     O 2.160000 1.220000 0.000000\n\
     Br -1.940000 0.000000 0.000000\n\
     H 0.220000 1.020000 0.000000\n\
     H 0.220000 -0.510000 0.884000\n\
     H 1.880000 -0.510000 -0.884000\n\
     H 1.880000 -0.510000 0.884000\n\
     H 2.960000 1.100000 0.500000\n";

fn energy(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    accelerator: SccAccelerator,
    level_shift: f64,
    scc_broyden: bool,
) -> f64 {
    let opts = ElectronicOptions {
        scc_accelerator: accelerator,
        level_shift,
        scc_broyden,
        energy_tolerance: 1.0e-9,
        charge_tolerance: 1.0e-8,
        max_scc: 1000,
        ..ElectronicOptions::default()
    };
    run_electronic(system, params, opts).unwrap().total_free
}

#[test]
fn accelerators_converge_to_same_energy() {
    let Some(params) = load_params() else {
        return;
    };
    for xyz in [WATER, BROMOETHANOL] {
        let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
        let reference = energy(&system, &params, SccAccelerator::Broyden, 0.0, true);
        for accelerator in [
            SccAccelerator::Linear,
            SccAccelerator::Cdiis,
            SccAccelerator::Newton,
        ] {
            let e = energy(&system, &params, accelerator, 0.0, true);
            assert!(
                (e - reference).abs() < 1.0e-7,
                "{accelerator:?}: energy {e} vs Broyden {reference} (diff {:.3e})",
                (e - reference).abs()
            );
        }
        // Level shifting must not change the converged energy.
        let e_shift = energy(&system, &params, SccAccelerator::Broyden, 0.15, true);
        assert!(
            (e_shift - reference).abs() < 1.0e-6,
            "level shift changed the energy: {e_shift} vs {reference}"
        );
        // CDIIS + level shift together.
        let e_both = energy(&system, &params, SccAccelerator::Cdiis, 0.1, true);
        assert!((e_both - reference).abs() < 1.0e-6);
    }
}
