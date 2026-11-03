// SPDX-License-Identifier: GPL-3.0-or-later
//! Full Cartesian finite-difference audits. No selected elements, contractions,
//! rigid-body projection or symmetrization of the numerical reference.
use gfn1_rs::linalg::Matrix;
use gfn1_rs::math::Vec3;
use gfn1_rs::soc::SocParameters;
use gfn1_rs::third_derivative::finite_t;
use gfn1_rs::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf, sync::Arc, time::Instant};

const WATER: &str = "3\nasymmetric water\nO .1 .2 0\nH .97 .74 .04\nH -.69 .83 -.03\n";
const HF: &str = "2\nskew HF\nF .1 -.2 .03\nH .8 .34 .43\n";
const PH3: &str =
    "4\nasymmetric phosphine\nP .1 .2 -.1\nH 1.24 .15 .72\nH -.5 1.22 .61\nH -.54 -.91 .66\n";
const DIAMOND: &str = "2\nLattice=\"0 1.7835 1.7835 1.7835 0 1.7835 1.7835 1.7835 0\" pbc=\"T T T\"\nC 0 0 0\nC .89175 .89175 .89175\n";
const PERIODIC_WATER: &str = "3\nLattice=\"3.8 0 0 .2 4 0 .1 .17 4.2\" pbc=\"T T T\"\nO .1 .2 0\nH .97 .74 .04\nH -.69 .83 -.03\n";

enum Route {
    Molecular { finite_t: bool },
    Periodic(PbcOptions),
    MolecularSoc(SocParameters),
    PeriodicSoc(PbcOptions, SocParameters),
    Magnetic(Option<SecondaryBasis>),
    Repulsion,
    Dispersion,
    Halogen,
}
struct Case {
    name: String,
    system: PeriodicSystem,
    parameters: Gfn1Parameters,
    options: ElectronicOptions,
    route: Route,
    maximum_order: u8,
}
fn options() -> ElectronicOptions {
    ElectronicOptions {
        electronic_temperature: 0.,
        energy_tolerance: 1e-13,
        charge_tolerance: 1e-11,
        max_scc: 1000,
        ..Default::default()
    }
}
fn case(name: &str, xyz: &str, maximum_order: u8) -> Case {
    Case {
        name: name.into(),
        system: PeriodicSystem::from_xyz_str(xyz, 0., false).unwrap(),
        parameters: Gfn1Parameters::builtin().unwrap(),
        options: options(),
        route: Route::Molecular { finite_t: false },
        maximum_order,
    }
}
fn coupling() -> SocParameters {
    SocParameters::from_zeta_str("units hartree\nprovenance Independent synthetic derivative-audit zeta; not literature constants\n8 1 .005\n").unwrap()
}
fn matrix_values(m: &Matrix) -> Vec<f64> {
    (0..m.rows())
        .flat_map(|i| (0..m.cols()).map(move |j| m[(i, j)]))
        .collect()
}
fn cubic_values(t: &third_derivative::SymmetricThird) -> Vec<f64> {
    let d = t.n();
    (0..d.pow(3))
        .map(|i| t.get(i / (d * d), i / d % d, i % d))
        .collect()
}
fn quartic_values(t: &fourth_derivative::SymmetricFourth) -> Vec<f64> {
    let d = t.n();
    (0..d.pow(4))
        .map(|i| t.get(i / d.pow(3), i / (d * d) % d, i / d % d, i % d))
        .collect()
}
fn slab_values(t: &[Matrix]) -> Vec<f64> {
    let d = t.len();
    (0..d.pow(3))
        .map(|i| t[i % d][(i / (d * d), i / d % d)])
        .collect()
}
fn gradient_values(g: Vec<Vec3>) -> Vec<f64> {
    g.into_iter().flat_map(|v| v.to_array()).collect()
}
fn invalid(message: impl Into<String>) -> Gfn1Error {
    Gfn1Error::InvalidInput(message.into())
}
fn check_state(e: &ElectronicResult, options: &ElectronicOptions) -> Result<()> {
    if !e.converged || (e.electronic_temperature - options.electronic_temperature).abs() > 1e-8 {
        return Err(invalid(format!(
            "FD audit state: converged={}, iterations={}, requested temperature={}, actual temperature={}",
            e.converged, e.iterations, options.electronic_temperature, e.electronic_temperature
        )));
    }
    Ok(())
}
fn evaluate(c: &Case, system: &PeriodicSystem, order: u8) -> Result<Vec<f64>> {
    let p = &c.parameters;
    let o = &c.options;
    let h = AnalyticHessianOptions {
        electronic_options: o.clone(),
        ..Default::default()
    };
    let cutoff = o.hamiltonian.coordination_cutoff;
    let values = match &c.route {
        Route::Molecular { finite_t: ft } => match order {
            0 => {
                let e = run_electronic(system, p, o.clone())?;
                check_state(&e, o)?;
                vec![e.total_free]
            }
            1 => {
                let g = analytic_gradient(
                    system,
                    p,
                    AnalyticGradientOptions {
                        electronic: o.clone(),
                        ..Default::default()
                    },
                )?;
                check_state(&g.electronic_result, o)?;
                gradient_values(g.gradient)
            }
            2 => matrix_values(&analytic_hessian(system, p, h)?.hessian),
            3 if *ft => cubic_values(&finite_t::third_derivative_finite_t_dense(
                system, p, &h, cutoff,
            )?),
            3 => cubic_values(&third_derivative_analytic_dense(system, p, h, cutoff)?),
            4 if *ft => quartic_values(&finite_t::fourth_derivative_finite_t_dense(
                system, p, &h, cutoff,
            )?),
            4 => quartic_values(&fourth_derivative_analytic_dense(system, p, &h, cutoff)?),
            _ => return Err(invalid("unsupported molecular audit order")),
        },
        Route::Periodic(b) => match order {
            0 => {
                let e = run_pbc_scc(system, p, o, b)?;
                if !e.converged {
                    return Err(invalid("unconverged periodic energy"));
                }
                vec![e.total_free]
            }
            1 => gradient_values(pbc_analytic_gradient(system, p, o, b)?.gradient),
            2 => matrix_values(
                &if b.kmesh.is_gamma_only() {
                    pbc_gamma_hessian(system, p, o, b)?
                } else {
                    pbc_kpoint_hessian(system, p, o, b)?
                }
                .hessian,
            ),
            3 => cubic_values(&if b.kmesh.is_gamma_only() {
                pbc_gamma_third_analytic_dense(system, p, o, b)?
            } else {
                pbc_kpoint_third_analytic_dense(system, p, o, b)?
            }),
            _ => return Err(invalid("periodic analytic FC4 has no implementation")),
        },
        Route::MolecularSoc(s) => match order {
            0 => vec![soc::run_soc_scc(system, p, o, s)?.total_free],
            1 => gradient_values(soc::soc_analytic_gradient(system, p, o, s)?.1),
            _ => {
                return Err(invalid(
                    "molecular SOC nuclear Hessian/FC3/FC4 has no public implementation",
                ))
            }
        },
        Route::PeriodicSoc(b, s) => match order {
            0 => vec![pbc::soc::run_pbc_soc_scc(system, p, o, b, s)?.total_free],
            1 => gradient_values(pbc::soc::pbc_soc_analytic_gradient(system, p, o, b, s)?.1),
            2 => matrix_values(&pbc::soc::pbc_soc_hessian(system, p, o, b, s)?.hessian),
            _ => {
                return Err(invalid(
                    "periodic SOC analytic FC3/FC4 has no implementation",
                ))
            }
        },
        Route::Magnetic(basis) => match order {
            0 => {
                let e = match basis {
                    Some(b) => run_magnetic_scc_m1(system, p, o, b)?,
                    None => run_magnetic_scc(system, p, o)?,
                };
                if !e.converged {
                    return Err(invalid("unconverged magnetic energy"));
                }
                vec![e.energy]
            }
            1 => gradient_values(
                magnetic_analytic_gradient(system, p, o, basis.as_ref(), 1e-4)?.gradient,
            ),
            _ => {
                return Err(invalid(
                    "magnetic nuclear Hessian/FC3/FC4 has no implementation",
                ))
            }
        },
        Route::Repulsion => match order {
            0 => vec![repulsion::repulsion_energy(system, p)?],
            1 => gradient_values(repulsion::repulsion_energy_gradient(system, p)?.gradient),
            2 => matrix_values(&repulsion::repulsion_energy_gradient_hessian(system, p)?.hessian),
            3 => slab_values(&repulsion::repulsion_third_derivative(system, p)?),
            4 => quartic_values(&repulsion::repulsion_fourth_derivative(system, p)?),
            _ => return Err(invalid("unsupported repulsion order")),
        },
        Route::Dispersion => match order {
            0 => vec![dispersion::dispersion_energy(system, p, None)?],
            1 => gradient_values(dispersion::dispersion_energy_gradient(system, p, None)?.gradient),
            2 => matrix_values(
                &dispersion::dispersion_energy_gradient_hessian(system, p, None)?.hessian,
            ),
            3 => dispersion::dispersion_third_derivative(system, p, None)?.third,
            4 => dispersion::dispersion_fourth_derivative(system, p, None)?.fourth,
            _ => return Err(invalid("unsupported dispersion order")),
        },
        Route::Halogen => match order {
            0 => vec![halogen::halogen_energy(system, p)?],
            1 => gradient_values(halogen::halogen_energy_gradient(system, p)?.gradient),
            2 => matrix_values(&halogen::halogen_energy_gradient_hessian(system, p)?.hessian),
            3 => slab_values(&halogen::halogen_third_derivative(system, p)?),
            4 => halogen::halogen_fourth_derivative(system, p)?.fourth,
            _ => return Err(invalid("unsupported halogen order")),
        },
    };
    let expected = (3 * system.atoms.len()).pow(u32::from(order));
    if values.len() != expected || values.iter().any(|v| !v.is_finite()) {
        return Err(invalid("non-finite or incomplete Cartesian tensor"));
    }
    Ok(values)
}
fn displaced(system: &PeriodicSystem, coordinate: usize, step: f64) -> PeriodicSystem {
    let mut out = system.clone();
    let p = &mut out.atoms[coordinate / 3].position;
    match coordinate % 3 {
        0 => p.x += step,
        1 => p.y += step,
        _ => p.z += step,
    }
    out
}
fn metrics(analytic: &[f64], numerical: &[f64], atol: f64, rtol: f64) -> Value {
    let mut maximum = 0_f64;
    let mut normalized = 0_f64;
    let mut worst = 0;
    let mut sum = 0.;
    let mut passed = 0;
    for (i, (&a, &b)) in analytic.iter().zip(numerical).enumerate() {
        let error = (a - b).abs();
        let ratio = error / (atol + rtol * a.abs().max(b.abs()));
        if error > maximum {
            maximum = error;
            worst = i;
        }
        normalized = normalized.max(ratio);
        sum += error * error;
        passed += usize::from(ratio <= 1.);
    }
    json!({"maximum_absolute_error":maximum,"rms_error":(sum / analytic.len() as f64).sqrt(),
           "maximum_error_over_local_tolerance":normalized,"worst_flat_index":worst,
           "passed_components":passed,"tested_components":analytic.len(),"pass":passed==analytic.len()})
}
fn save(path: &std::path::Path, value: &Value) -> Result<()> {
    fs::write(
        path,
        serde_json::to_vec_pretty(value).map_err(|e| invalid(e.to_string()))?,
    )?;
    Ok(())
}
fn audit(c: &Case, path: &std::path::Path) -> Result<Value> {
    let started = Instant::now();
    let d = 3 * c.system.atoms.len();
    let atoms: Vec<_> = c
        .system
        .atoms
        .iter()
        .map(|a| json!({"Z":a.z,"position_bohr":a.position.to_array()}))
        .collect();
    let mut report = json!({"schema":"gfn1-rs-full-cartesian-derivative-v1","status":"running",
        "package_version":env!("CARGO_PKG_VERSION"),"dependency_lockfile_sha256":format!("{:x}",Sha256::digest(include_bytes!("../Cargo.lock"))),"audit_driver_sha256":format!("{:x}",Sha256::digest(include_bytes!("nuclear_derivative_audit.rs"))),"case":c.name,"ndof":d,"atoms":atoms,
        "analytic_orders_requested":c.maximum_order,"unsupported_higher_orders":((c.maximum_order+1)..=4).collect::<Vec<_>>(),
        "energy_definition":"stationary Mermin free energy (classical component for term-only cases)",
        "electronic_temperature_K":c.options.electronic_temperature,"scf_energy_tolerance":c.options.energy_tolerance,
        "scf_charge_tolerance":c.options.charge_tolerance,"parameter_sha256":format!("{:x}",Sha256::digest(c.parameters.to_param_string().as_bytes())),
        "index_convention":"row-major Cartesian indices; DOF=3*atom+axis; final index is differentiated coordinate; no FD symmetrization or projection",
        "orders":[]});
    if let Route::Molecular { finite_t } = c.route {
        let e = run_electronic(&c.system, &c.parameters, c.options.clone())?;
        check_state(&e, &c.options)?;
        let fractional = e
            .occupations
            .iter()
            .filter(|&&v| v > 1e-8 && v < 2. - 1e-8)
            .count();
        if finite_t && fractional == 0 {
            return Err(invalid(
                "finite-T audit did not exercise fractional occupations",
            ));
        }
        report["reference_occupations"] = json!(e.occupations);
        report["fractional_orbitals"] = json!(fractional);
    }
    let neighborhood = match &c.route {
        Route::Periodic(b) | Route::PeriodicSoc(b, _) => Some(
            smoothness::NuclearNeighborhood::periodic_scalar(&c.system, &c.options, b)?,
        ),
        _ => None,
    };
    save(path, &report)?;
    let mut all_passed = true;
    for order in 1..=c.maximum_order {
        println!(
            "cartesian_audit {} order={order} components={}",
            c.name,
            d.pow(u32::from(order))
        );
        let analytic = evaluate(c, &c.system, order)?;
        let size = analytic.len();
        let base_step = if order <= 2 { 4e-4 } else { 2e-3 };
        let (atol, rtol) = match order {
            1 => (1e-6, 2e-5),
            2 => (3e-6, 3e-5),
            3 => (5e-6, 5e-5),
            _ => (2e-5, 1e-4),
        };
        let mut steps = Vec::new();
        let mut references = Vec::new();
        for step in [base_step, base_step / 2., base_step / 4.] {
            let mut numerical = vec![0.; size];
            for coordinate in 0..d {
                let plus = displaced(&c.system, coordinate, step);
                let minus = displaced(&c.system, coordinate, -step);
                if let Some(domain) = &neighborhood {
                    if !domain.changed_terms(&plus)?.is_empty()
                        || !domain.changed_terms(&minus)?.is_empty()
                    {
                        return Err(invalid(format!("finite difference crossed a nuclear cutoff branch: order={order} coordinate={coordinate} step={step:e}")));
                    }
                }
                let a = evaluate(c, &plus, order - 1)?;
                let b = evaluate(c, &minus, order - 1)?;
                for i in 0..a.len() {
                    numerical[i * d + coordinate] = (a[i] - b[i]) / (2. * step);
                }
            }
            let error = metrics(&analytic, &numerical, atol, rtol);
            println!(
                "cartesian_audit {} order={order} h={step:e} max_error={} passed={}/{}",
                c.name, error["maximum_absolute_error"], error["passed_components"], size
            );
            steps.push(json!({"step_bohr":step,"metrics":error,"finite_difference":numerical}));
            references.push(numerical);
        }
        let extrapolated: Vec<_> = references[2]
            .iter()
            .zip(&references[1])
            .map(|(a, b)| (4. * a - b) / 3.)
            .collect();
        let richardson = metrics(&analytic, &extrapolated, atol, rtol);
        let passed = steps[2]["metrics"]["pass"] == true && richardson["pass"] == true;
        all_passed &= passed;
        report["orders"].as_array_mut().unwrap().push(json!({"order":order,"status":if passed{"pass"}else{"fail"},
            "components":size,"absolute_tolerance":atol,"relative_tolerance":rtol,"analytic":analytic,"steps":steps,
            "richardson_metrics":richardson,"richardson_finite_difference":extrapolated}));
        save(path, &report)?;
    }
    report["status"] = json!(if all_passed { "pass" } else { "fail" });
    report["seconds"] = json!(started.elapsed().as_secs_f64());
    save(path, &report)?;
    Ok(report)
}
fn run_group(name: &str, mut cases: Vec<Case>) {
    let selected = std::env::var("GFN1_DERIVATIVE_AUDIT_CASE").ok();
    if let Some(id) = &selected {
        cases.retain(|c| c.name == *id);
    }
    assert!(!cases.is_empty(), "zero matching audit cases");
    let root = std::env::var_os("GFN1_VALIDATION_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/validation/nuclear-derivatives")
        });
    fs::create_dir_all(&root).unwrap();
    let mut summary = Vec::new();
    for c in cases {
        let path = root.join(format!("{}.json", c.name));
        let result = audit(&c, &path).unwrap_or_else(|error| {
            let mut record = fs::read(&path)
                .ok()
                .and_then(|s| serde_json::from_slice::<Value>(&s).ok())
                .unwrap_or_else(|| json!({"case":c.name,"orders":[]}));
            record["status"] = json!("calculation_error");
            record["error"] = json!(error.to_string());
            save(&path, &record).unwrap();
            eprintln!("cartesian_audit_error {}: {error}", c.name);
            record
        });
        summary.push(json!({"case":c.name,"status":result["status"],"maximum_order":c.maximum_order,
            "orders":result["orders"].as_array().unwrap().iter().map(|o|json!({"order":o["order"],"status":o["status"],
                "components":o["components"],"fine_metrics":o["steps"][2]["metrics"],"richardson_metrics":o["richardson_metrics"]})).collect::<Vec<_>>(),
            "details_file":format!("{}.json",c.name)}));
    }
    let passed = summary.iter().filter(|r| r["status"] == "pass").count();
    save(&root.join(format!("group-{name}.json")), &json!({"schema":"gfn1-rs-full-cartesian-group-v1",
        "package_version":env!("CARGO_PKG_VERSION"),"scope":if selected.is_some(){"diagnostic_subset"}else{"full_group"},
        "executed_cases":summary.len(),"passed_cases":passed,"status":if passed==summary.len(){"pass"}else{"fail"},"cases":summary})).unwrap();
    assert_eq!(
        passed,
        summary.len(),
        "one or more full-component derivative audits failed"
    );
}

#[test]
#[ignore = "extended full Cartesian ranks 1..4, three finite-difference steps"]
fn full_cartesian_molecular_ladders() {
    let mut cases = vec![
        case("molecule-water-t0", WATER, 4),
        case("molecule-hf-t0", HF, 4),
    ];
    let mut warm = case("molecule-water-fermi", WATER, 4);
    warm.route = Route::Molecular { finite_t: true };
    warm.options.electronic_temperature = 10000.;
    cases.push(warm);
    let mut warm_hf = case("molecule-hf-fermi-degenerate", HF, 4);
    warm_hf.route = Route::Molecular { finite_t: true };
    warm_hf.options.electronic_temperature = 30000.;
    cases.push(warm_hf);
    for (name, xyz, model) in [
        (
            "ml-h2-fci",
            "2\nskew H2\nH .1 -.2 .03\nH .72 .21 -.15\n",
            "builtin:h2-fci",
        ),
        ("ml-water-clusters", WATER, "builtin:water-clusters"),
    ] {
        let mut c = case(name, xyz, 3);
        c.options.ml_model = Some(Arc::new(ml::MlHamiltonian::resolve(model).unwrap()));
        cases.push(c);
    }
    run_group("molecular", cases);
}

#[test]
#[ignore = "all Cartesian energy/gradient/Hessian/FC3/FC4 components of classical terms"]
fn full_cartesian_classical_components() {
    let mut rep = case("component-repulsion", WATER, 4);
    rep.route = Route::Repulsion;
    let mut d3 = case("component-d3-two-body", WATER, 4);
    d3.route = Route::Dispersion;
    let mut atm = case("component-d3-with-atm", WATER, 4);
    atm.route = Route::Dispersion;
    atm.parameters.globpar.insert("s9".into(), 1.);
    let mut hal = case(
        "component-halogen",
        "3\nhalogen-contact BrCN geometry\nC 0 0 0\nBr 1.94 .12 -.04\nN 4.65 .31 .17\n",
        4,
    );
    hal.route = Route::Halogen;
    assert!(
        evaluate(&hal, &hal.system, 0).unwrap()[0].abs() > 1e-10,
        "halogen audit must be active"
    );
    run_group("components", vec![rep, d3, atm, hal]);
}

#[test]
#[ignore = "full-coordinate gradients of solvent, spin, SOC, fields and experimental models"]
fn full_cartesian_extended_gradients() {
    let mut cases = Vec::new();
    for (name, model, solvent) in [
        ("alpb-water", SolvationModel::Alpb, "water"),
        ("gbsa-water", SolvationModel::Gbsa, "water"),
        ("alpb-benzene", SolvationModel::Alpb, "benzene"),
        ("gbsa-acetonitrile", SolvationModel::Gbsa, "acetonitrile"),
    ] {
        let mut c = case(name, WATER, 1);
        c.options.solvation = Some(SolvationOptions::new(model, solvent));
        cases.push(c);
    }
    for (name, xyz, multiplicity) in [
        ("spin-oh", "2\nOH\nO .1 -.2 .03\nH .72 .5 -.21\n", 2),
        ("spin-o2", "2\nO2\nO .1 -.2 .03\nO .91 .63 -.29\n", 3),
    ] {
        let mut c = case(name, xyz, 1);
        c.options.spin_polarization = true;
        c.options.spin_multiplicity = Some(multiplicity);
        c.options.electronic_temperature = 300.;
        cases.push(c);
    }
    let mut soc = case("soc-molecular-water", WATER, 1);
    soc.route = Route::MolecularSoc(coupling());
    cases.push(soc);
    for (name, basis) in [
        ("magnetic-m0", None),
        (
            "magnetic-m1",
            Some(builtin_secondary("cc-pVDZ").unwrap().unwrap()),
        ),
    ] {
        let mut c = case(name, WATER, 1);
        c.options.external_field.magnetic_field = Some(Vec3::new(0.003, -0.005, 0.007));
        c.route = Route::Magnetic(basis);
        cases.push(c);
    }
    let mut field = case("electric-field", WATER, 1);
    field.options.external_field.electric_field = Some(Vec3::new(0.002, -0.003, 0.001));
    cases.push(field);
    let mut d4 = case("experimental-d4", WATER, 1);
    d4.options.experimental_d4 = true;
    cases.push(d4);
    let mut mp = case("multipole-mdftb2", WATER, 1);
    mp.options.multipole = true;
    cases.push(mp);
    let mut camm = case("multipole-camm", WATER, 1);
    camm.options.multipole = true;
    camm.options.multipole_model = MultipoleModel::CammOnMdftb2;
    cases.push(camm);
    let mut oct = case("multipole-octupole", PH3, 1);
    oct.options.multipole = true;
    oct.options.multipole_octupole = true;
    cases.push(oct);
    let mut rank4 = case(
        "multipole-rank4-hcl-t0",
        "2\nasymmetric HCl\nCl 0 0 0\nH .3 .2 1.25\n",
        1,
    );
    rank4.options.multipole = true;
    rank4.options.multipole_order = 4;
    cases.push(rank4);
    let mut ph3_rank4 = case("multipole-rank4-ph3-t0", PH3, 1);
    ph3_rank4.options.multipole = true;
    ph3_rank4.options.multipole_order = 4;
    cases.push(ph3_rank4);
    let mut cross = case("multipole-charge-cross", WATER, 1);
    cross.options.multipole = true;
    cross.options.multipole_order = 2;
    cross.options.multipole_charge_order = vec![3, 4];
    cases.push(cross);
    for order in [4, 6] {
        let mut c = case(&format!("charge-order-{order}"), WATER, 1);
        c.options.charge_order = order;
        cases.push(c);
    }
    for (name, onsite) in [("exchange-mfx", false), ("exchange-ofx", true)] {
        let mut c = case(name, WATER, 1);
        c.options.lr_exchange = true;
        c.options.onsite_exchange = onsite;
        cases.push(c);
    }
    let mut u = case(
        "fixed-hubbard-u",
        "2\nskew SH doublet\nS .1 -.2 .03\nH .33 .27 1.25\n",
        1,
    );
    u.options.spin_polarization = true;
    u.options.spin_multiplicity = Some(2);
    u.options.electronic_temperature = 300.;
    u.options.plus_u = true;
    u.options.hubbard_u = vec![(16, 0.5)];
    cases.push(u);
    let mut uv = case(
        "fixed-hubbard-uv",
        "2\nskew S2 triplet\nS .1 -.2 .03\nS 1.21 .73 1.08\n",
        1,
    );
    uv.options.spin_polarization = true;
    uv.options.spin_multiplicity = Some(3);
    uv.options.electronic_temperature = 300.;
    uv.options.plus_u = true;
    uv.options.plus_u_v = true;
    uv.options.hubbard_u = vec![(16, 0.2)];
    uv.options.hubbard_v = vec![(16, 16, 0.05)];
    uv.options.hubbard_v_cutoff = 6.;
    cases.push(uv);
    run_group("extensions", cases);
}

#[test]
#[ignore = "full Cartesian periodic scalar/ML/SOC ladders at Gamma and multiple k points"]
fn full_cartesian_periodic_ladders() {
    let mut cases = Vec::new();
    for (label, mesh) in [
        ("gamma", KMesh::gamma()),
        ("k211", KMesh::monkhorst_pack([2, 1, 1])),
    ] {
        let pbc = PbcOptions {
            kmesh: mesh,
            ao_cutoff: 20.7,
            ewald: EwaldOptions {
                real_cutoff: 30.23,
                sr_cutoff: 8.17,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut plain = case(&format!("periodic-scalar-{label}"), DIAMOND, 3);
        plain.route = Route::Periodic(PbcOptions {
            ao_cutoff: 12.37,
            ..pbc.clone()
        });
        cases.push(plain);
        let mut warm = case(&format!("periodic-scalar-fermi-{label}"), PERIODIC_WATER, 2);
        warm.route = Route::Periodic(pbc.clone());
        warm.options.electronic_temperature = 10000.;
        cases.push(warm);
        let mut ml = case(&format!("periodic-ml-{label}"), PERIODIC_WATER, 2);
        ml.route = Route::Periodic(pbc.clone());
        ml.options.ml_model = Some(Arc::new(
            ml::MlHamiltonian::resolve("builtin:water-clusters").unwrap(),
        ));
        cases.push(ml);
        for temperature in [0., 10000.] {
            let mut soc = case(
                &format!("periodic-soc-{label}-t{temperature:.0}"),
                PERIODIC_WATER,
                2,
            );
            soc.route = Route::PeriodicSoc(pbc.clone(), coupling());
            soc.options.electronic_temperature = temperature;
            cases.push(soc);
        }
    }
    run_group("periodic", cases);
}
