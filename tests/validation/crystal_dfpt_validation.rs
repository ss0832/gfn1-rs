// SPDX-License-Identifier: GPL-3.0-or-later
//! Developer-only validation/diagnostic driver; see tests/README.md.
//! Fully specified crystal inputs and every fixed-cell Hessian FD component.
use gfn1_rs::lattice::Lattice;
use gfn1_rs::linalg::{symmetric_eigen, Matrix};
use gfn1_rs::math::Vec3;
use gfn1_rs::pbc::{
    pbc_analytic_gradient, pbc_gamma_hessian, pbc_gradient_from_scc, pbc_kpoint_hessian, KMesh,
    PbcOptions,
};
use gfn1_rs::{Atom, ElectronicOptions, Gfn1Parameters, PeriodicSystem};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{path::PathBuf, time::Instant};
fn matrix(m: &Matrix) -> Vec<Vec<f64>> {
    (0..m.rows())
        .map(|i| (0..m.cols()).map(|j| m[(i, j)]).collect())
        .collect()
}
fn calculate(
    name: &str,
    z: [u8; 2],
    mass: [f64; 2],
    a: f64,
    tetrahedral: bool,
    mesh: [usize; 3],
    params: &Gfn1Parameters,
) -> gfn1_rs::Result<Value> {
    let start = Instant::now();
    let b = gfn1_rs::constants::ANGSTROM_TO_BOHR;
    let lattice = Lattice::from_vectors(
        Vec3::new(0., a / 2., a / 2.) * b,
        Vec3::new(a / 2., 0., a / 2.) * b,
        Vec3::new(a / 2., a / 2., 0.) * b,
        [true; 3],
    )?;
    let r = if tetrahedral {
        Vec3::new(a / 4., a / 4., a / 4.)
    } else {
        Vec3::new(a / 2., 0., 0.)
    };
    let system = PeriodicSystem::new(
        vec![
            Atom {
                z: z[0],
                position: Vec3::zero(),
            },
            Atom {
                z: z[1],
                position: r * b,
            },
        ],
        Some(lattice),
    );
    let mut electronic = ElectronicOptions {
        electronic_temperature: 300.,
        energy_tolerance: 1e-13,
        charge_tolerance: 1e-11,
        max_scc: 1000,
        mixing: 0.1,
        ..Default::default()
    };
    let pbc = PbcOptions {
        kmesh: KMesh {
            size: mesh,
            gamma_centered: true,
            fold_time_reversal: true,
        },
        ..Default::default()
    };
    let mut solved = None;
    let mut attempts = Vec::new();
    for mixing in [0.4, 0.1, 0.05] {
        electronic.mixing = mixing;
        let attempt = if mesh == [1; 3] {
            pbc_gamma_hessian(&system, params, &electronic, &pbc)
        } else {
            pbc_kpoint_hessian(&system, params, &electronic, &pbc)
        };
        match attempt {
            Ok(result) => {
                attempts.push(json!({"mixing":mixing,"status":"converged"}));
                solved = Some(result);
                break;
            }
            Err(error) if error.to_string().contains("SCC did not converge") => attempts
                .push(json!({"mixing":mixing,"status":"not_converged","error":error.to_string()})),
            Err(error) => return Err(error),
        }
    }
    let result = solved.ok_or_else(|| {
        gfn1_rs::Gfn1Error::InvalidInput(format!(
            "SCC failed at all recorded mixing settings: {attempts:?}"
        ))
    })?;
    if !result.scf.converged {
        return Err(gfn1_rs::Gfn1Error::InvalidInput(
            "crystal SCC failed to converge".into(),
        ));
    }
    let energy = result.scf.total_free;
    let temperature = result.scf.electronic_temperature;
    let iterations = result.scf.iterations;
    let ground = pbc_gradient_from_scc(&system, params, result.scf, &electronic, &pbc)?;
    let hessian = result.hessian;
    let n = 6;
    let mut steps = Vec::new();
    for h in [1e-3, 5e-4] {
        let mut fd = Matrix::zeros(n, n);
        for coordinate in 0..n {
            let gradient = |displacement: f64| -> gfn1_rs::Result<Vec<Vec3>> {
                let mut s = system.clone();
                let r = &mut s.atoms[coordinate / 3].position;
                match coordinate % 3 {
                    0 => r.x += displacement,
                    1 => r.y += displacement,
                    _ => r.z += displacement,
                }
                Ok(pbc_analytic_gradient(&s, params, &electronic, &pbc)?.gradient)
            };
            let plus = gradient(h)?;
            let minus = gradient(-h)?;
            for row in 0..n {
                fd[(row, coordinate)] = (plus[row / 3].to_array()[row % 3]
                    - minus[row / 3].to_array()[row % 3])
                    / (2. * h);
            }
        }
        let error = hessian
            .as_slice()
            .iter()
            .zip(fd.as_slice())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        steps.push((h, fd, error));
    }
    let mut max_error = 0.0_f64;
    let mut asr = 0.0_f64;
    let mut richardson = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let fd = (4. * steps[1].1[(i, j)] - steps[0].1[(i, j)]) / 3.;
            richardson[(i, j)] = fd;
            max_error = max_error.max((hessian[(i, j)] - fd).abs());
        }
    }
    for row in 0..n {
        for axis in 0..3 {
            asr = asr.max((hessian[(row, axis)] + hessian[(row, 3 + axis)]).abs());
        }
    }
    let mut dynamical = hessian.clone();
    for i in 0..n {
        for j in 0..n {
            dynamical[(i, j)] /= (mass[i / 3] * mass[j / 3]).sqrt();
        }
    }
    let frequencies: Vec<_> = symmetric_eigen(&dynamical)?
        .values
        .iter()
        .map(|x| {
            x.signum()
                * (x.abs() / gfn1_rs::constants::AMU_TO_ELECTRON_MASS).sqrt()
                * gfn1_rs::constants::HARTREE_TO_WAVENUMBER
        })
        .collect();
    let status =
        if max_error < 2e-5 && asr < 1e-8 && result.raw_max_asymmetry < 1e-8 && temperature == 300.
        {
            "pass"
        } else {
            "fail"
        };
    eprintln!("{name} {mesh:?}: {status}, 36-component FD error={max_error:e}, ASR={asr:e}, frequencies={frequencies:?}");
    Ok(
        json!({"name":name,"status":status,"structure":if tetrahedral {"diamond/zincblende primitive fcc"}else{"rocksalt primitive fcc"},
        "lattice_constant_angstrom":a,"numbers":z,"masses_amu":mass,"positions_angstrom":[[0.,0.,0.],r.to_array()],
        "cell_angstrom":[[0.,a/2.,a/2.],[a/2.,0.,a/2.],[a/2.,a/2.,0.]],"charge":0,"electronic_temperature_kelvin":temperature,"kmesh":mesh,"electronic_mesh_gamma_centered":true,
        "geometry_status":"specified representative input, not optimized or asserted as an experimental reference", "energy_free_hartree_per_primitive_cell":energy,
        "scc_iterations":iterations,"scc_converged":true,"scc_mixing":electronic.mixing,"scc_broyden":electronic.scc_broyden,"scc_attempts":attempts,"raw_hessian_asymmetry_hartree_per_bohr2":result.raw_max_asymmetry,
        "numerical_controls":{"energy_tolerance_hartree":electronic.energy_tolerance,"charge_tolerance":electronic.charge_tolerance,"max_scc":electronic.max_scc,"pbc_options":format!("{pbc:?}")},
        "hessian_hartree_per_bohr2":matrix(&hessian),"max_abs_gradient_hartree_per_bohr":ground.gradient.iter().flat_map(|g|g.to_array()).map(f64::abs).fold(0.0_f64,f64::max),
        "gradient_hartree_per_bohr":ground.gradient.iter().map(|g|g.to_array()).collect::<Vec<_>>(),
        "asr_max_hartree_per_bohr2":asr,"fd_steps_bohr":[steps[0].0,steps[1].0],"fd_max_errors":[steps[0].2,steps[1].2],"richardson_max_error":max_error,
        "fd_hessians_hartree_per_bohr2":[matrix(&steps[0].1),matrix(&steps[1].1)],"richardson_hessian_hartree_per_bohr2":matrix(&richardson),
        "tested_hessian_components":36,"gamma_mode_wavenumbers_cm1":frequencies,"elapsed_seconds":start.elapsed().as_secs_f64()}),
    )
}
fn main() -> gfn1_rs::Result<()> {
    let params = Gfn1Parameters::builtin()?;
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/validation/v070/crystal-dfpt-validation.json"));
    let mut cases = Vec::new();
    for (name, z, mass, a, tetrahedral) in [
        ("C-diamond", [6, 6], [12.011, 12.011], 3.567, true),
        ("Si-diamond", [14, 14], [28.085, 28.085], 5.431, true),
        ("SiC-zincblende", [14, 6], [28.085, 12.011], 4.36, true),
        ("LiF-rocksalt", [3, 9], [6.94, 18.998], 4.026, false),
        ("NaCl-rocksalt", [11, 17], [22.990, 35.45], 5.64, false),
        ("MgO-rocksalt", [12, 8], [24.305, 15.999], 4.212, false),
    ] {
        for mesh in [[1, 1, 1], [2, 2, 2]] {
            let result=calculate(name,z,mass,a,tetrahedral,mesh,&params).unwrap_or_else(|e| {eprintln!("{name} {mesh:?}: ERROR {e}");json!({"name":name,"kmesh":mesh,"status":"calculation_error","error":e.to_string()})});
            cases.push(result);
            let report = json!({"schema":"gfn1-rs-crystal-dfpt-validation-v1","package_version":env!("CARGO_PKG_VERSION"),"electronic_temperature_kelvin":300,"geometry_protocol":"explicit representative lattices; no geometry or cell optimization; no experimental-accuracy claim",
                "source_sha256":{"tests/validation/crystal_dfpt_validation.rs":format!("{:x}",Sha256::digest(include_bytes!("crystal_dfpt_validation.rs"))),"src/pbc/hessian.rs":format!("{:x}",Sha256::digest(include_bytes!("../../src/pbc/hessian.rs"))),"Cargo.lock":format!("{:x}",Sha256::digest(include_bytes!("../../Cargo.lock")))},"cases":cases});
            if let Some(parent) = output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&output, serde_json::to_vec_pretty(&report).unwrap())?;
        }
    }
    if cases.iter().any(|c| c["status"] != "pass") {
        return Err(gfn1_rs::Gfn1Error::InvalidInput(
            "one or more crystal DFPT gates failed; complete results retained".into(),
        ));
    }
    Ok(())
}
