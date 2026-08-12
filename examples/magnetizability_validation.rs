// SPDX-License-Identifier: GPL-3.0-or-later
//! GFN1-xTB-M0 / GFN1-xTB-M1 isotropic magnetizability validation against the
//! reference values of Cheng & Wibowo-Teale, *J. Chem. Theory Comput.* **19**,
//! 6226 (2023), SI Table 2 (10^-30 J/T^2). The benchmark geometries are the
//! standard equilibrium structures of the Lutnaes et al. magnetizability set
//! (*J. Chem. Phys.* **131**, 144104 (2009)); experimental/near-exact bond
//! lengths and angles are used here.
//!
//! Run (GFN1_XTB_PARAM is optional; the builtin parameters are used when unset):
//!   GFN1_M1_BASIS=/path/GFN1-xTB-cc-pVDZ.txt \
//!   cargo run --release --example magnetizability_validation
//!
//! M1 (node-correct dual basis) is the production magnetic method and should
//! track the paper's M1 column closely. M0 (single nodeless minimal basis) is
//! the paper's baseline and is intrinsically sensitive to the basis kinetic
//! energy, so it is reported only to illustrate the M0 -> M1 improvement.

use gfn1_rs::math::Vec3;
use gfn1_rs::{
    magnetizability_isotropic, parse_secondary_basis, ElectronicOptions, ExternalFieldOptions,
    Gfn1Parameters, PeriodicSystem, SecondaryBasis, MAGNETIZABILITY_AU_TO_SI,
};

/// (name, xyz, paper M0, paper M1, paper CCSD(T)) in 10^-30 J/T^2.
struct Case {
    name: &'static str,
    xyz: String,
    paper_m0: f64,
    paper_m1: f64,
    ccsdt: f64,
}

fn diatomic(a: &str, b: &str, r: f64) -> String {
    format!("2\n{a}{b}\n{a} 0.0 0.0 0.0\n{b} {r} 0.0 0.0\n")
}

fn cases() -> Vec<Case> {
    // Experimental / near-exact equilibrium geometries (Angstrom).
    let water = {
        // r(OH) = 0.9578, angle(HOH) = 104.48 deg, in the xy-plane.
        let r = 0.9578_f64;
        let half = 104.48_f64.to_radians() / 2.0;
        let (x, y) = (r * half.sin(), r * half.cos());
        format!(
            "3\nH2O\nO 0.0 0.0 0.0\nH {x:.4} {y:.4} 0.0\nH {:.4} {y:.4} 0.0\n",
            -x
        )
    };
    let nh3 = {
        // r(NH) = 1.0124, angle(HNH) = 106.67 deg -> tilt beta from C3 axis.
        let r = 1.0124_f64;
        let cos_hnh = 106.67_f64.to_radians().cos();
        let cos2b = (cos_hnh + 0.5) / 1.5; // cos(HNH) = cos^2 b - 0.5 sin^2 b
        let cosb = cos2b.sqrt();
        let sinb = (1.0 - cos2b).sqrt();
        let z = r * cosb;
        let rho = r * sinb;
        let mut s = String::from("4\nNH3\nN 0.0 0.0 0.0\n");
        for k in 0..3 {
            let phi = (k as f64) * 120.0_f64.to_radians();
            s.push_str(&format!(
                "H {:.4} {:.4} {z:.4}\n",
                rho * phi.cos(),
                rho * phi.sin()
            ));
        }
        s
    };
    let ch4 = {
        // r(CH) = 1.0870, tetrahedral.
        let d = 1.0870_f64 / 3.0_f64.sqrt();
        format!(
            "5\nCH4\nC 0.0 0.0 0.0\nH {d:.4} {d:.4} {d:.4}\nH {d:.4} {:.4} {:.4}\nH {:.4} {d:.4} {:.4}\nH {:.4} {:.4} {d:.4}\n",
            -d, -d, -d, -d, -d, -d
        )
    };
    vec![
        Case {
            name: "HF",
            xyz: diatomic("H", "F", 0.9168),
            paper_m0: -133.0,
            paper_m1: -147.2,
            ccsdt: -176.4,
        },
        Case {
            name: "CO",
            // The CO bond length in Å coincidentally matches the first digits
            // of 2/√π — it is a bond length, not the math constant.
            #[allow(clippy::approx_constant)]
            xyz: diatomic("C", "O", 1.1283),
            paper_m0: -94.0,
            paper_m1: -197.6,
            ccsdt: -209.5,
        },
        Case {
            name: "N2",
            xyz: diatomic("N", "N", 1.0977),
            paper_m0: -69.9,
            paper_m1: -217.7,
            ccsdt: -205.2,
        },
        Case {
            name: "H2O",
            xyz: water,
            paper_m0: -168.2,
            paper_m1: -201.7,
            ccsdt: -235.1,
        },
        Case {
            name: "NH3",
            xyz: nh3,
            paper_m0: -189.9,
            paper_m1: -257.6,
            ccsdt: -290.3,
        },
        Case {
            name: "CH4",
            xyz: ch4,
            paper_m0: -249.0,
            paper_m1: -342.4,
            ccsdt: -316.9,
        },
    ]
}

fn main() {
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    let basis_path = std::env::var("GFN1_M1_BASIS")
        .expect("set GFN1_M1_BASIS to the GFN1-xTB-cc-pVDZ secondary-basis file");
    let secondary: SecondaryBasis =
        parse_secondary_basis(&std::fs::read_to_string(basis_path).unwrap()).unwrap();

    let base = ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1.0e-10,
        charge_tolerance: 1.0e-9,
        external_field: ExternalFieldOptions {
            magnetic_field: Some(Vec3::zero()),
            ..ExternalFieldOptions::default()
        },
        ..ElectronicOptions::default()
    };
    let step: f64 = std::env::var("GFN1_MAG_STEP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.02);
    eprintln!("(finite-field step = {step} a.u.)");

    println!("Isotropic magnetizability (10^-30 J/T^2) vs Cheng & Wibowo-Teale 2023 SI Table 2");
    println!(
        "{:<5} {:>9} {:>9} | {:>9} {:>9} {:>9} | {:>8}",
        "mol", "M0(this)", "M1(this)", "M0(ref)", "M1(ref)", "CCSD(T)", "M1 err%"
    );
    let mut max_m1_err = 0.0_f64;
    let mut sum_abs_err = 0.0_f64;
    let mut count = 0usize;
    for case in cases() {
        let system = PeriodicSystem::from_xyz_str(&case.xyz, 0.0, false).unwrap();
        let m0 = magnetizability_isotropic(&system, &params, &base, None, step).unwrap()
            * MAGNETIZABILITY_AU_TO_SI;
        let m1 = magnetizability_isotropic(&system, &params, &base, Some(&secondary), step)
            .unwrap()
            * MAGNETIZABILITY_AU_TO_SI;
        let err_pct = 100.0 * (m1 - case.paper_m1).abs() / case.paper_m1.abs();
        max_m1_err = max_m1_err.max(err_pct);
        sum_abs_err += (m1 - case.paper_m1).abs();
        count += 1;
        println!(
            "{:<5} {:>9.1} {:>9.1} | {:>9.1} {:>9.1} {:>9.1} | {:>7.1}%",
            case.name, m0, m1, case.paper_m0, case.paper_m1, case.ccsdt, err_pct
        );
    }
    println!(
        "\nM1 vs paper: max error {:.1}%, mean |delta| {:.1} x10^-30 J/T^2 over {} molecules",
        max_m1_err,
        sum_abs_err / count as f64,
        count
    );
}
