// SPDX-License-Identifier: GPL-3.0-or-later
//! Same-temperature, same-threshold cold SCC benchmark; failures remain failures.
use gfn1_rs::pbc::scf::run_pbc_scc;
use gfn1_rs::pbc::{KMesh, PbcOptions};
use gfn1_rs::resources::{ResourceContext, ResourceLimits};
use gfn1_rs::{ElectronicOptions, Gfn1Parameters, PeriodicSystem, SccAccelerator};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output = std::env::args().nth(1).ok_or("output JSON path required")?;
    let reference: Value = serde_json::from_str(include_str!("../data/crystal-dfpt-results.json"))?;
    let mut inputs = reference["cases"].as_array().unwrap().clone();
    let a = (435.9234894069041_f64 / 2.).cbrt();
    let b = gfn1_rs::constants::BOHR_TO_ANGSTROM;
    for scale in [0.94_f64, 1., 1.06] {
        let r = a * scale.cbrt() * b;
        inputs.push(json!({"name":format!("LiF-expanded-volume-{scale}"),"numbers":[3,9],
            "cell_angstrom":[[0.,r,r],[r,0.,r],[r,r,0.]],"positions_angstrom":[[0.,0.,0.],[r,0.,0.]],
            "kmesh":[1,1,1],"electronic_mesh_gamma_centered":true}));
    }
    let context = ResourceContext::new(ResourceLimits::from_megabytes(Some(2), Some(2048))?)?;
    let params = Gfn1Parameters::builtin()?;
    let mut cases = Vec::new();
    for input in &inputs {
        let matrix = &input["cell_angstrom"];
        let flat: Vec<_> = matrix
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
            .collect();
        let mut xyz = format!(
            "{}\nLattice=\"{}\" pbc=\"T T T\"\n",
            input["numbers"].as_array().unwrap().len(),
            flat.iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        );
        for (z, p) in input["numbers"]
            .as_array()
            .unwrap()
            .iter()
            .zip(input["positions_angstrom"].as_array().unwrap())
        {
            xyz += &format!(
                "{} {} {} {}\n",
                gfn1_rs::system::z_to_symbol(z.as_u64().unwrap() as u8).unwrap(),
                p[0],
                p[1],
                p[2]
            );
        }
        let system = PeriodicSystem::from_xyz_str(&xyz, 0., false)?;
        let mesh: [usize; 3] =
            std::array::from_fn(|i| input["kmesh"][i].as_u64().unwrap() as usize);
        let pbc = PbcOptions {
            ao_cutoff: 30.,
            kmesh: KMesh {
                size: mesh,
                gamma_centered: true,
                fold_time_reversal: true,
            },
            ..Default::default()
        };
        let mut methods = Vec::new();
        for (name, method) in [
            ("auto", SccAccelerator::Auto),
            ("broyden", SccAccelerator::Broyden),
            ("newton", SccAccelerator::Newton),
            ("cdiis", SccAccelerator::Cdiis),
            ("linear", SccAccelerator::Linear),
        ] {
            let electronic = ElectronicOptions {
                electronic_temperature: 300.,
                max_scc: 1000,
                energy_tolerance: 1e-13,
                charge_tolerance: 1e-11,
                mixing: 0.4,
                scc_accelerator: method,
                ..Default::default()
            };
            let start = Instant::now();
            let result = context.run(|| run_pbc_scc(&system, &params, &electronic, &pbc));
            let elapsed = start.elapsed().as_secs_f64();
            let value = match result {
                Ok(s) => {
                    json!({"method":name,"status":"converged","elapsed_seconds":elapsed,"iterations":s.iterations,
                    "free_energy_hartree":s.total_free,"shell_charges":s.shell_charges,"diagnostics":s.scc_diagnostics.as_json(),
                    "reference_energy_difference_hartree":input["energy_free_hartree_per_primitive_cell"].as_f64().map(|e|s.total_free-e)})
                }
                Err(e) => {
                    json!({"method":name,"status":"failed","elapsed_seconds":elapsed,"error":e.to_string()})
                }
            };
            eprintln!(
                "{} {mesh:?} {name}: {} ({elapsed:.3}s)",
                input["name"], value["status"]
            );
            methods.push(value);
        }
        cases.push(json!({"name":input["name"],"kmesh":mesh,"gamma_centered":true,"geometry":input,"methods":methods}));
        let hashes: serde_json::Map<_, _> = [
            "src/pbc/scf.rs",
            "src/pbc/scc_solver.rs",
            "src/electronic.rs",
            "tests/validation/pbc_scc_validation.rs",
        ]
        .into_iter()
        .map(|path| {
            Ok((
                path.to_string(),
                json!(format!("{:x}", Sha256::digest(std::fs::read(path)?))),
            ))
        })
        .collect::<Result<_, std::io::Error>>()?;
        std::fs::write(
            &output,
            serde_json::to_vec_pretty(
                &json!({"schema":"gfn1-rs-periodic-scc-benchmark-v1","package_version":env!("CARGO_PKG_VERSION"),
            "electronic_temperature_kelvin":300.,"mixing":0.4,"charge_tolerance":1e-11,"energy_tolerance_hartree":1e-13,"max_scc":1000,
            "ao_cutoff_bohr":30.,"initial_guess":"cold, uniform charge-conserving shell guess","resources":{"threads":2,"workspace_memory_mb":2048},
            "source_sha256":hashes,"cases":cases}),
            )?,
        )?;
    }
    Ok(())
}
