// SPDX-License-Identifier: GPL-3.0-or-later
//! Developer-only validation/diagnostic driver; see tests/README.md.
//! Real molecular PES-only VPT2, independent normal-force-field differences,
//! and the closed one-dimensional perturbation formula. No experimental fit.
use gfn1_rs::constants::HARTREE_TO_WAVENUMBER;
use gfn1_rs::linalg::Matrix;
use gfn1_rs::math::Vec3;
use gfn1_rs::*;
use serde_json::{json, Value};
use sha2::Digest;

fn contract(a: &[f64], h: &Matrix, b: &[f64]) -> f64 {
    (0..a.len())
        .flat_map(|i| (0..b.len()).map(move |j| a[i] * h[(i, j)] * b[j]))
        .sum()
}
fn displace(system: &PeriodicSystem, direction: &[f64], h: f64) -> PeriodicSystem {
    let mut out = system.clone();
    for (atom, v) in out.atoms.iter_mut().zip(direction.chunks_exact(3)) {
        atom.position += Vec3::new(v[0], v[1], v[2]) * h;
    }
    out
}
fn close(a: &[f64], b: &[f64], atol: f64, rtol: f64) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(a, b)| (a - b).abs() <= atol + rtol * a.abs().max(b.abs()))
}

fn calculate(
    name: &str,
    xyz: &str,
    isotope: Option<f64>,
    p: &Gfn1Parameters,
    o: &ElectronicOptions,
) -> Result<Value> {
    let initial = PeriodicSystem::from_xyz_str(xyz, 0., false)?;
    let optimized = optimize_geometry(
        &initial,
        p,
        GeometryOptimizationOptions {
            gradient_tolerance: 1e-8,
            gradient_options: AnalyticGradientOptions {
                electronic: o.clone(),
                ..Default::default()
            },
            ..Default::default()
        },
    )?;
    assert!(optimized.converged);
    let system = &optimized.system;
    let masses: Vec<_> = system
        .atoms
        .iter()
        .map(|a| isotope.unwrap_or_else(|| data_tables::relative_atomic_mass(a.z)))
        .collect();
    let r = molecular_vpt2_pes(system, p, o.clone(), &masses, Default::default(), 1e-7)?;
    let nm = &r.normal_modes;
    let directions = nm.atomic_unit_directions();
    let m = directions.len();
    let ho = AnalyticHessianOptions {
        electronic_options: o.clone(),
        ..Default::default()
    };
    let base = analytic_hessian(system, p, ho.clone())?.hessian;
    let cubic: Vec<_> = (0..m.pow(3))
        .map(|x| r.force_field.cubic.get(x / (m * m), x / m % m, x % m))
        .collect();
    let quartic: Vec<_> = (0..m * m)
        .map(|x| r.force_field.quartic_semidiagonal[(x / m, x % m)])
        .collect();
    let mut c_refs = Vec::new();
    let mut q_refs = Vec::new();
    let mut steps = Vec::new();
    for h in [0.04_f64, 0.02, 0.01] {
        let mut c = vec![0.; m.pow(3)];
        let mut q = vec![0.; m * m];
        for (k, direction) in directions.iter().enumerate() {
            let plus = analytic_hessian(&displace(system, direction, h), p, ho.clone())?.hessian;
            let minus = analytic_hessian(&displace(system, direction, -h), p, ho.clone())?.hessian;
            for i in 0..m {
                for j in 0..m {
                    c[(i * m + j) * m + k] = (contract(&directions[i], &plus, &directions[j])
                        - contract(&directions[i], &minus, &directions[j]))
                        / (2. * h);
                }
                q[i * m + k] = (contract(&directions[i], &plus, &directions[i])
                    - 2. * contract(&directions[i], &base, &directions[i])
                    + contract(&directions[i], &minus, &directions[i]))
                    / h.powi(2);
            }
        }
        steps.push(json!({"normal_coordinate_step_au":h,"cubic_fd":c,"quartic_semidiagonal_fd":q}));
        c_refs.push(c);
        q_refs.push(q);
    }
    let rich = |v: &[Vec<f64>]| -> Vec<f64> {
        v[2].iter()
            .zip(&v[1])
            .map(|(f, c)| (4. * f - c) / 3.)
            .collect()
    };
    let cr = rich(&c_refs);
    let qr = rich(&q_refs);
    let cubic_pass = close(&cubic, &c_refs[2], 1e-10, 5e-5) && close(&cubic, &cr, 1e-10, 5e-5);
    let quartic_pass =
        close(&quartic, &q_refs[2], 1e-10, 1e-3) && close(&quartic, &qr, 1e-10, 1e-3);
    let independent_1d = if m == 1 {
        let w = r.force_field.omega[0];
        let c = cubic[0];
        let q = quartic[0];
        let ground = 0.5 * w + q / (32. * w.powi(2)) - 11. * c * c / (288. * w.powi(4));
        let fundamental =
            (w + q / (8. * w.powi(2)) - 5. * c * c / (24. * w.powi(4))) * HARTREE_TO_WAVENUMBER;
        assert!((ground - r.spectrum.ground.energy_hartree()).abs() < 1e-13);
        assert!((fundamental - r.spectrum.fundamentals_cm1[0]).abs() < 1e-8);
        json!({"status":"pass","ground_hartree":ground,"fundamental_cm1":fundamental,
            "formula":"omega + Phi4/(8 omega^2) - 5 Phi3^2/(24 omega^4)"})
    } else {
        Value::Null
    };
    let positive = r
        .spectrum
        .fundamentals_cm1
        .iter()
        .all(|x| x.is_finite() && *x > 0.);
    let positions: Vec<_> = system
        .atoms
        .iter()
        .map(|a| json!({"Z":a.z,"position_bohr":a.position.to_array()}))
        .collect();
    println!("VPT2 {name}: harmonic={:?} fundamental={:?} cubic_fd={cubic_pass} quartic_fd={quartic_pass}",nm.vibrations.wavenumbers,r.spectrum.fundamentals_cm1);
    Ok(
        json!({"case":name,"status":if positive && cubic_pass && quartic_pass {"pass"}else{"fail"},
        "atoms":positions,"masses_amu":masses,"vibrational_modes":m,"rigid_directions":nm.rigid_directions.len(),
        "residual_gradient_hartree_per_bohr":r.residual_gradient_hartree_per_bohr,
        "harmonic_cm1":nm.vibrations.wavenumbers,"fundamental_cm1":r.spectrum.fundamentals_cm1,
        "ground_hartree":r.spectrum.ground.energy_hartree(),"ground_cubic_correction_hartree":r.spectrum.ground.cubic_correction_hartree,
        "ground_quartic_correction_hartree":r.spectrum.ground.quartic_correction_hartree,
        "omega_au":r.force_field.omega,"normal_cubic":cubic,"normal_quartic_semidiagonal":quartic,
        "finite_difference_ladder":steps,"cubic_richardson":cr,"quartic_richardson":qr,
        "cubic_fd_pass":cubic_pass,"quartic_fd_pass":quartic_pass,"independent_one_dimensional_formula":independent_1d}),
    )
}
fn main() -> Result<()> {
    let p = Gfn1Parameters::builtin()?;
    let o = ElectronicOptions {
        electronic_temperature: 0.,
        energy_tolerance: 1e-13,
        charge_tolerance: 1e-11,
        max_scc: 1000,
        ..Default::default()
    };
    let mut cases = Vec::new();
    for (name, xyz, isotope) in [
        ("H2", include_str!("../../examples/h2.xyz"), None),
        (
            "D2",
            include_str!("../../examples/h2.xyz"),
            Some(2.01410177812),
        ),
        ("HF", "2\nHF\nF 0 0 0\nH .3 .2 .85\n", None),
        ("H2O", include_str!("../../examples/water.xyz"), None),
    ] {
        cases.push(calculate(name, xyz, isotope, &p, &o)?);
    }
    let passed = cases.iter().filter(|c| c["status"] == "pass").count();
    let report = json!({"schema":"gfn1-rs-molecular-vpt2-audit-v1","package_version":env!("CARGO_PKG_VERSION"),"dependency_lockfile_sha256":format!("{:x}",sha2::Sha256::digest(include_bytes!("../../Cargo.lock"))),
        "scope":"PES-only VPT2; no Watson/Coriolis kinetic terms or anharmonic intensities; model-internal validation, not experimental frequency accuracy",
        "reference":"Independent normal-coordinate Hessian differences for all normal cubic and semidiagonal quartic components; closed one-dimensional perturbation formula",
        "electronic_temperature_K":0,"resonance_threshold_cm1":10,"minimum_frequency_cm1":20,
        "passed_cases":passed,"executed_cases":cases.len(),"status":if passed==cases.len(){"pass"}else{"fail"},"cases":cases});
    let root = std::path::PathBuf::from(
        std::env::var("GFN1_VALIDATION_DIR").unwrap_or_else(|_| "target/validation/vpt2".into()),
    );
    std::fs::create_dir_all(&root)?;
    std::fs::write(
        root.join("vpt2-molecular-audit.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )?;
    assert_eq!(passed, cases.len(), "VPT2 audit failed");
    Ok(())
}
