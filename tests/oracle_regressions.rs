// SPDX-License-Identifier: GPL-3.0-or-later
use gfn1_rs::*;
#[test]
fn asymmetric_water_dimer_zero_overlap_transverse_forces_match_energy() {
    let system=PeriodicSystem::from_xyz_str("6\nwater dimer\nO 0 0 0\nH .96 0 0\nH -.24 .93 0\nO 2.75 .1 0\nH 3.1 .9 0\nH 3.3 -.65 .1\n",0.0,false).unwrap();
    let params = Gfn1Parameters::builtin().unwrap();
    let options = ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1e-12,
        charge_tolerance: 1e-10,
        ..Default::default()
    };
    let gradient = analytic_gradient(
        &system,
        &params,
        AnalyticGradientOptions {
            electronic: options.clone(),
            ..Default::default()
        },
    )
    .unwrap();
    for (atom, axis) in [(0, 1), (1, 1), (3, 2), (4, 2)] {
        for h in [2e-4, 1e-4] {
            let energy = |sign: f64| {
                let mut x = system.clone();
                let r = &mut x.atoms[atom].position;
                match axis {
                    0 => r.x += sign * h,
                    1 => r.y += sign * h,
                    _ => r.z += sign * h,
                };
                run_electronic(&x, &params, options.clone())
                    .unwrap()
                    .total_free
            };
            let fd = (energy(1.0) - energy(-1.0)) / (2.0 * h);
            let actual = gradient.gradient[atom].to_array()[axis];
            eprintln!("atom={atom} axis={axis}: analytic {actual:e} FD {fd:e}, electronic {:?}, rep {:?}, disp {:?}",gradient.electronic_gradient[atom],gradient.repulsion_gradient[atom],gradient.dispersion_gradient[atom]);
            assert!((actual - fd).abs() < 1e-6);
        }
    }
}
