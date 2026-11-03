// SPDX-License-Identifier: GPL-3.0-or-later
//! JSONL probe of the native orbital/CI model for independent PES diagnostics.
use gfn1_rs::stda_xtb::{
    solve_stda_xtb_native, stda_xtb_orbitals_at_temperature, NativeStdaOptions,
};
use gfn1_rs::{PeriodicSystem, Result};
use std::io::{BufRead, Write};
fn system(data: &serde_json::Value) -> PeriodicSystem {
    let mut xyz = String::from("6\nseam\n");
    for (i, z) in data["numbers"].as_array().unwrap().iter().enumerate() {
        let r = &data["positions_angstrom"][i];
        xyz.push_str(&format!(
            "{} {} {} {}\n",
            gfn1_rs::z_to_symbol(z.as_u64().unwrap() as u8).unwrap(),
            r[0],
            r[1],
            r[2]
        ));
    }
    PeriodicSystem::from_xyz_str(&xyz, 0., false).unwrap()
}
fn main() -> Result<()> {
    let path = std::env::args().nth(1).expect("pass a reference JSON path");
    let file: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?).unwrap();
    let reference = &file["reference_geometry"];
    let base = stda_xtb_orbitals_at_temperature(&system(&reference), 512, 300.)?;
    for line in std::io::stdin().lock().lines() {
        let input: serde_json::Value = serde_json::from_str(&line?).unwrap();
        let sys = system(&input);
        let orbital = stda_xtb_orbitals_at_temperature(&sys, 512, 300.)?;
        let no = orbital.nocc;
        let metric = base
            .coefficients
            .transpose()
            .matmul(&orbital.overlap)?
            .matmul(&orbital.coefficients)?;
        let overlap = vec![
            vec![metric[(no - 1, no - 1)], metric[(no - 1, no)]],
            vec![metric[(no, no - 1)], metric[(no, no)]],
        ];
        let energies = vec![
            orbital.energies[no - 1],
            orbital.energies[no] - 3.1 / 27.21139570,
        ];
        let mut result = serde_json::json!({"overlap":overlap,"energies_hartree":energies,"charges":orbital.vtb_charges});
        if input["spectrum"].as_bool().unwrap_or(false) {
            let response = solve_stda_xtb_native(
                &sys,
                &NativeStdaOptions {
                    electronic_temperature: 300.,
                    energy_window_ev: input["energy_window_ev"].as_f64().unwrap_or(20.),
                    n_states: 400,
                    ..Default::default()
                },
            )?;
            result["ci_eigenvalues_hartree"] = serde_json::json!(response.ci_eigenvalues_hartree);
            result["configurations"] = serde_json::json!(response.configurations);
            result["primary_configuration_count"] =
                serde_json::json!(response.primary_configuration_count);
            result["promoted_configuration_count"] =
                serde_json::json!(response.promoted_configuration_count);
        }
        println!("{result}");
        std::io::stdout().flush()?;
    }
    Ok(())
}
