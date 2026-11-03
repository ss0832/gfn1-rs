// SPDX-License-Identifier: GPL-3.0-or-later
use gfn1_rs::pbc::scf::{run_pbc_scc, run_pbc_scc_with_guess, PbcSccGuess};
use gfn1_rs::pbc::{pbc_gradient_from_scc, pbc_stress_from_scc, KMesh};
use gfn1_rs::{ElectronicOptions, Gfn1Parameters, PbcOptions, PeriodicSystem, SccAccelerator};

fn lif(volume: f64) -> PeriodicSystem {
    let a = (volume / 2.).cbrt();
    PeriodicSystem::from_xyz_str(
        &format!(
            "2\nLattice=\"0 {a} {a} {a} 0 {a} {a} {a} 0\" pbc=\"T T T\"\nLi 0 0 0\nF {a} 0 0\n"
        ),
        0.,
        true,
    )
    .unwrap()
}

#[test]
fn cold_lif_newton_auto_and_warm_reference_same_free_energy_forces_and_stress() {
    let params = Gfn1Parameters::builtin().unwrap();
    let pbc = PbcOptions {
        ao_cutoff: 30.,
        ..Default::default()
    };
    let base = ElectronicOptions {
        electronic_temperature: 300.,
        energy_tolerance: 1e-13,
        charge_tolerance: 1e-11,
        max_scc: 120,
        mixing: 0.1,
        ..Default::default()
    };
    for volume in [
        435.9234894069041,
        435.9234894069041 * 0.94,
        435.9234894069041 * 1.06,
    ] {
        let system = lif(volume);
        let now = std::time::Instant::now();
        let direct = run_pbc_scc(
            &system,
            &params,
            &ElectronicOptions {
                scc_accelerator: SccAccelerator::Newton,
                ..base.clone()
            },
            &pbc,
        )
        .unwrap();
        let auto = run_pbc_scc(&system, &params, &base, &pbc).unwrap();
        let guess = PbcSccGuess::from(&direct);
        let warm = run_pbc_scc_with_guess(
            &system,
            &params,
            &ElectronicOptions {
                scc_accelerator: SccAccelerator::Broyden,
                ..base.clone()
            },
            &pbc,
            Some(&guess),
        )
        .unwrap();
        eprintln!("LiF V={volume} Newton {} maps {}, Auto {} maps {}, warm {} E={:.15} elapsed {:.3}s diagnostics {}",direct.iterations,direct.scc_diagnostics.map_evaluations,auto.iterations,auto.scc_diagnostics.map_evaluations,warm.iterations,direct.total_free,now.elapsed().as_secs_f64(),auto.scc_diagnostics.as_json());
        assert!((direct.total_free - auto.total_free).abs() < 2e-11);
        assert!((direct.total_free - warm.total_free).abs() < 2e-11);
        if (volume - 435.9234894069041).abs() < 1e-8 {
            assert!((direct.total_free + 5.4464146416367125).abs() < 2e-10);
        }
        assert!(direct.iterations < 50);
        assert!(auto.iterations < 100);
        assert!(direct.scc_diagnostics.newton_accepted > 0);
        assert!((direct.shell_charges.iter().sum::<f64>()).abs() < 1e-10);
        let gd = pbc_gradient_from_scc(&system, &params, direct.clone(), &base, &pbc).unwrap();
        let ga = pbc_gradient_from_scc(&system, &params, auto.clone(), &base, &pbc).unwrap();
        for (x, y) in gd.gradient.iter().zip(&ga.gradient) {
            assert!((*x - *y).norm() < 1e-8);
        }
        let sd = pbc_stress_from_scc(
            &system,
            &params,
            direct.clone(),
            &base,
            &pbc,
            system.lattice.as_ref().unwrap(),
        )
        .unwrap();
        let sa = pbc_stress_from_scc(
            &system,
            &params,
            auto.clone(),
            &base,
            &pbc,
            system.lattice.as_ref().unwrap(),
        )
        .unwrap();
        assert!(sd
            .stress
            .as_slice()
            .iter()
            .zip(sa.stress.as_slice())
            .all(|(x, y)| (x - y).abs() < 1e-9));
    }
}

#[test]
fn explicit_mixers_and_ml_mesh_preserve_same_fixed_point() {
    let system = PeriodicSystem::from_xyz_str(
        "2\nLattice=\"4 .2 .1 0 4.5 .3 .1 0 5\" pbc=\"T T T\"\nH .2 .3 .1\nF 1.1 .6 .4\n",
        0.,
        false,
    )
    .unwrap();
    let params = Gfn1Parameters::builtin().unwrap();
    let mut model = gfn1_rs::ml::MlHamiltonian::zero(vec![1, 9], vec![1.3, 3., 5.], 8.);
    model.heads.get_mut(&9).unwrap().levels[1].weights[1] = 0.008;
    let pbc = PbcOptions {
        ao_cutoff: 9.,
        kmesh: KMesh {
            size: [2, 2, 2],
            gamma_centered: false,
            fold_time_reversal: false,
        },
        ..Default::default()
    };
    let mut reference: Option<f64> = None;
    for method in [
        SccAccelerator::Linear,
        SccAccelerator::Broyden,
        SccAccelerator::Cdiis,
        SccAccelerator::Newton,
        SccAccelerator::Auto,
    ] {
        let opts = ElectronicOptions {
            electronic_temperature: 30000.,
            energy_tolerance: 1e-13,
            charge_tolerance: 1e-11,
            max_scc: 500,
            mixing: 0.1,
            scc_accelerator: method,
            ml_model: Some(std::sync::Arc::new(model.clone())),
            ..Default::default()
        };
        let result = run_pbc_scc(&system, &params, &opts, &pbc).unwrap();
        eprintln!(
            "ML HF mesh {:?} {method:?}: {} iterations E={} {}",
            pbc.kmesh.size,
            result.iterations,
            result.total_free,
            result.scc_diagnostics.as_json()
        );
        if let Some(energy) = reference {
            assert!((result.total_free - energy).abs() < 2e-11);
        } else {
            reference = Some(result.total_free);
        }
        assert!(result.ml_parameters.is_some());
    }
}
