// SPDX-License-Identifier: GPL-3.0-or-later
//! Developer-only validation/diagnostic driver; see tests/README.md.
//! Same-input, 300 K comparison and an exact CI expectation-value decomposition.
use gfn1_rs::stda_xtb::{solve_stda_xtb, NativeStdaOptions};
use gfn1_rs::{
    run_electronic, solve_tda, solve_tda_with_dipole, ElectronicOptions, Gfn1Parameters,
    PeriodicSystem, TdaOptions, TdaTransitionDipole,
};
use serde_json::json;
fn main() -> gfn1_rs::Result<()> {
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("../data/tda-stda-native-comparison.json"))
            .expect("comparison input");
    let params = Gfn1Parameters::builtin()?;
    let mut cases = Vec::new();
    for case in reference["cases"].as_array().expect("cases") {
        let numbers = case["numbers"].as_array().expect("numbers");
        let positions = case["positions_angstrom"].as_array().expect("positions");
        let mut xyz = format!("{}\nsame input, angstrom\n", numbers.len());
        for (z, r) in numbers.iter().zip(positions) {
            xyz.push_str(&format!(
                "{} {} {} {}\n",
                gfn1_rs::z_to_symbol(z.as_u64().unwrap() as u8).expect("element"),
                r[0],
                r[1],
                r[2]
            ));
        }
        let system = PeriodicSystem::from_xyz_str(&xyz, 0., false)?;
        let options = ElectronicOptions {
            electronic_temperature: 300.,
            energy_tolerance: 1e-13,
            charge_tolerance: 1e-11,
            max_scc: 1000,
            ..Default::default()
        };
        let ground = run_electronic(&system, &params, options.clone())?;
        let tda = solve_tda(
            &system,
            &params,
            &ground,
            TdaOptions {
                n_states: 10,
                ..Default::default()
            },
        )?;
        let eps = gfn1_rs::linalg::lowdin_solve_generalized(
            &ground.fock,
            &ground.integrals.overlap,
            1e-12,
        )?
        .values;
        let tda_ao = solve_tda_with_dipole(
            &system,
            &params,
            &ground,
            TdaOptions {
                n_states: 10,
                ..Default::default()
            },
            TdaTransitionDipole::AoLength,
        )?;
        let stda_options = NativeStdaOptions {
            energy_window_ev: 20.,
            n_states: 100,
            electronic_temperature: 300.,
            ..Default::default()
        };
        let stda = solve_stda_xtb(&system, &stda_options)?;
        let tda0 = solve_tda(
            &system,
            &params,
            &run_electronic(
                &system,
                &params,
                ElectronicOptions {
                    electronic_temperature: 0.,
                    ..options
                },
            )?,
            TdaOptions {
                n_states: 10,
                ..Default::default()
            },
        )?;
        let stda0 = solve_stda_xtb(
            &system,
            &NativeStdaOptions {
                electronic_temperature: 0.,
                ..stda_options
            },
        )?;
        let factor = gfn1_rs::constants::HARTREE_TO_EV;
        let tda_parts:Vec<_>=tda.states.iter().map(|s| {
            let gap=tda.pairs.iter().zip(&s.amplitudes).map(|(&(i,a),x)|x*x*(eps[a]-eps[i])).sum::<f64>();
            json!({"energy_ev":s.excitation_energy*factor,"gap_expectation_ev":gap*factor,"response_expectation_ev":(s.excitation_energy-gap)*factor})
        }).collect();
        let stda_parts:Vec<_>=stda.states.iter().map(|s| {
            let gap=stda.configurations.iter().zip(&s.amplitudes).map(|(&(i,a),x)|x*x*(stda.orbital_energies_hartree[a]-stda.orbital_energies_hartree[i])).sum::<f64>();
            json!({"energy_ev":s.excitation_ev,"gap_expectation_ev":gap*27.21139570,"response_selection_expectation_ev":s.excitation_ev-gap*27.21139570,"virtual_shift_ev":3.1,"unshifted_gap_expectation_ev":gap*27.21139570-3.1})
        }).collect();
        let occupation_error = ground
            .occupations
            .iter()
            .map(|f| f.abs().min((f - 2.).abs()))
            .fold(0.0_f64, f64::max);
        let states = |result: &gfn1_rs::TdaResult| {
            result.states.iter().enumerate().map(|(i,s)|json!({"root":i+1,"excitation_ev":s.excitation_energy*factor,"oscillator_strength":s.oscillator_strength})).collect::<Vec<_>>()
        };
        cases.push(json!({"name":case["name"],"numbers":numbers,"positions_angstrom":positions,"charge":0,"spin":"singlet","electronic_temperature_kelvin":300,
            "tda":states(&tda),"tda_ao_length":states(&tda_ao),"stda_xtb":stda.states.iter().enumerate().map(|(i,s)|json!({"root":i+1,"excitation_ev":s.excitation_ev,"oscillator_strength":s.oscillator_strength})).collect::<Vec<_>>(),
            "tda_basis_size":ground.basis.len(),"stda_basis_size":stda.basis_size,
            "tda_maximum_occupation_deviation":occupation_error,"stda_maximum_occupation_deviation":stda.maximum_occupation_deviation,
            "tda_max_0_to_300K_excitation_change_ev":tda.states.iter().zip(&tda0.states).map(|(a,b)|(a.excitation_energy-b.excitation_energy).abs()*factor).fold(0.0_f64,f64::max),
            "stda_max_0_to_300K_excitation_change_ev":stda.states.iter().zip(&stda0.states).map(|(a,b)|(a.excitation_ev-b.excitation_ev).abs()).fold(0.0_f64,f64::max),
            "tda_decomposition":tda_parts,"stda_decomposition":stda_parts}));
    }
    println!("{}",serde_json::to_string_pretty(&json!({"schema":"gfn1-rs-tda-stda-decomposition-v1","package_version":env!("CARGO_PKG_VERSION"),"parameter_sha256":gfn1_rs::stda_xtb::parameter_sha256(),"temperature_kelvin":300,"cases":cases})).unwrap());
    Ok(())
}
