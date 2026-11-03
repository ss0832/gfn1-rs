// SPDX-License-Identifier: GPL-3.0-or-later
use gfn1_rs::cli::properties::{execute_request, operation_catalog};
use gfn1_rs::resources::{ResourceContext, ResourceLimits};
use gfn1_rs::{ElectronicOptions, Gfn1Parameters, PeriodicSystem};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn context() -> ResourceContext {
    ResourceContext::new(ResourceLimits::from_megabytes(Some(2), Some(512)).unwrap()).unwrap()
}
fn request() -> Value {
    json!({"schema":"gfn1-rs-crystal-properties-v1",
    "geometry":{"numbers":[1,1],"positions":[[0.2,0.3,0.1],[1.6,0.45,0.2]],"cell":[[5.6,0.3,0.],[0.,6.1,0.2],[0.1,0.,6.4]],"coordinate_units":"bohr","charge":0},
    "electronic":{"temperature_kelvin":300.,"energy_tolerance_hartree":1e-13,"charge_tolerance":1e-11},
    "periodic":{"kmesh":[1,1,1],"gamma_centered":true,"fold_time_reversal":false,"ao_cutoff_bohr":20.},"tasks":[]})
}
fn task(id: &str, op: &str, args: Value) -> Value {
    json!({"id":id,"operation":op,"arguments":args})
}
fn result<'a>(report: &'a Value, id: &str) -> &'a Value {
    &report["tasks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["id"] == id)
        .unwrap()["result"]
}
fn max_difference(a: &Value, b: &Value) -> f64 {
    match (a, b) {
        (Value::Array(x), Value::Array(y)) => {
            assert_eq!(x.len(), y.len());
            x.iter()
                .zip(y)
                .map(|(x, y)| max_difference(x, y))
                .fold(0., f64::max)
        }
        _ => (a.as_f64().unwrap() - b.as_f64().unwrap()).abs(),
    }
}
fn temporary() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "gfn1-crystal-cli-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn conditional_native_ml_calculations_and_result_chains_match_typed_apis() {
    let dir = temporary();
    std::fs::write(
        dir.join("builtin-user.txt"),
        Gfn1Parameters::builtin().unwrap().to_param_string(),
    )
    .unwrap();
    let mut model = gfn1_rs::ml::MlHamiltonian::zero(vec![1], vec![1.2, 3.5, 5.8], 9.);
    model.heads.get_mut(&1).unwrap().levels[0].weights = vec![0.003, -0.0025, 0.004];
    model.heads.get_mut(&1).unwrap().correction.weights = vec![0.0004, 0.0002, -0.0003];
    std::fs::write(
        dir.join("user weights.model"),
        model.to_model_string().unwrap(),
    )
    .unwrap();
    let mut request = request();
    request["ml_model_path"] = json!("user weights.model");
    request["parameter_file"] = json!("builtin-user.txt");
    request["tasks"] = json!([
        task("ground", "ground_state", json!({})),
        task("h", "hessian", json!({})),
        task("c", "elastic_constants", json!({})),
        task("p", "berry_polarization", json!({})),
        task("z", "born_effective_charges", json!({})),
        task("eps", "dielectric_tensor", json!({})),
        task("coupled", "electromechanical_response", json!({})),
        task("ph", "phonon_mesh", json!({"repetitions":[1,1,1]})),
        task(
            "q",
            "phonon_qpoints",
            json!({"repetitions":[1,1,1],"q_fractional":[[0.,0.,0.]],"q_weights":[1.]})
        ),
        task(
            "bands",
            "band_structure",
            json!({"kpoints_fractional":[[0.,0.,0.],[0.2,0.,0.],[0.3,0.1,0.]]})
        ),
        task(
            "ed",
            "electronic_dos",
            json!({"energies_hartree":{"$ref":"bands#/energies_hartree"},"weights":[0.2,0.3,0.5],"grid_ev":[-40.,-30.,-20.,-10.,0.,10.,20.]})
        ),
        task(
            "pd",
            "phonon_dos",
            json!({"wavenumbers_cm1":{"$ref":"ph#/wavenumbers_cm1"},"weights":{"$ref":"ph#/weights"},"grid_cm1":[-5000.,-1000.,0.,1000.,5000.]})
        ),
        task(
            "ir",
            "polar_mode_properties",
            json!({"hessian_hartree_per_bohr2":{"$ref":"h#/hessian_hartree_per_bohr2"},"born_effective_charges":{"$ref":"z#/born_effective_charges"},"volume_bohr3":{"$ref":"z#/volume_bohr3"},"masses_amu":[1.00794,1.00794]})
        ),
        task(
            "lo",
            "lo_to_modes",
            json!({"hessian_hartree_per_bohr2":{"$ref":"h#/hessian_hartree_per_bohr2"},"born_effective_charges":{"$ref":"z#/born_effective_charges"},"electronic_relative_dielectric":{"$ref":"eps#/electronic_relative_dielectric"},"volume_bohr3":{"$ref":"z#/volume_bohr3"},"masses_amu":[1.00794,1.00794],"direction":[1.,0.,0.]})
        )
    ]);
    let report = context().run(|| execute_request(&request, &dir)).unwrap();
    assert_eq!(report["status"], "completed", "{}", report);
    assert_eq!(report["conditions"]["electronic_temperature_kelvin"], 300.);
    assert!(report["parameters"]
        .as_str()
        .unwrap()
        .contains("builtin-user.txt"));
    assert_eq!(
        report["ml_model_identity"]["canonical_sha256"],
        model.canonical_sha256().unwrap()
    );
    let system = PeriodicSystem::from_xyz_str(
        "2\nLattice=\"5.6 .3 0 0 6.1 .2 .1 0 6.4\" pbc=\"T T T\"\nH .2 .3 .1\nH 1.6 .45 .2\n",
        0.,
        true,
    )
    .unwrap();
    let opts = ElectronicOptions {
        electronic_temperature: 300.,
        energy_tolerance: 1e-13,
        charge_tolerance: 1e-11,
        ml_model: Some(Arc::new(model)),
        ..Default::default()
    };
    let pbc = gfn1_rs::pbc::PbcOptions {
        ao_cutoff: 20.,
        kmesh: gfn1_rs::pbc::KMesh {
            size: [1; 3],
            gamma_centered: true,
            fold_time_reversal: false,
        },
        ..Default::default()
    };
    let (scf, bands, elastic) = context()
        .run(|| {
            let params = Gfn1Parameters::from_file(dir.join("builtin-user.txt"))?;
            Ok((
                gfn1_rs::pbc::run_pbc_scc(&system, &params, &opts, &pbc)?,
                gfn1_rs::pbc::spectra::pbc_band_structure(
                    &system,
                    &params,
                    &opts,
                    &pbc,
                    &[[0., 0., 0.], [0.2, 0., 0.], [0.3, 0.1, 0.]],
                )?,
                gfn1_rs::pbc::elastic_response::pbc_elastic_response(
                    &system,
                    &params,
                    &opts,
                    &pbc,
                    &Default::default(),
                )?,
            ))
        })
        .unwrap();
    assert!(
        (result(&report, "ground")["energy_hartree"]
            .as_f64()
            .unwrap()
            - scf.total_free)
            .abs()
            < 1e-12
    );
    assert!(
        max_difference(
            &result(&report, "bands")["energies_hartree"],
            &bands.as_json()["energies_hartree"]
        ) < 1e-12
    );
    assert!(
        max_difference(
            &result(&report, "c")["clamped_stiffness_hartree_per_bohr3"],
            &elastic.as_json()["clamped_stiffness_hartree_per_bohr3"]
        ) < 1e-12
    );
    assert!(
        max_difference(
            &result(&report, "ph")["wavenumbers_cm1"],
            &result(&report, "q")["wavenumbers_cm1"]
        ) < 1e-10
    );
    assert_eq!(
        result(&report, "coupled")["piezoelectric_strain_au"],
        Value::Null,
        "nonstationary model must retain unavailable d"
    );
    assert!(report["tasks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["id"] == "ir")
        .unwrap()["provenance"]
        .get("ml_model_identity")
        .is_none());
}

#[test]
fn relaxation_geometry_references_and_real_eos_use_the_declared_conditions() {
    let mut request = request();
    let a = 6.0181095893;
    request["geometry"] = json!({"numbers":[3,9],"positions":[[0.,0.,0.],[a,0.,0.]],"cell":[[0.,a,a],[a,0.,a],[a,a,0.]],"coordinate_units":"bohr","charge":0.});
    request["periodic"]["ao_cutoff_bohr"] = json!(30.);
    request["tasks"] = json!([
        task(
            "relax",
            "relax_periodic",
            json!({"max_iterations":30,"force_tolerance_hartree_per_bohr":1e-7})
        ),
        json!({"id":"at_relaxed","operation":"ground_state","geometry":{"numbers":{"$ref":"relax#/numbers"},"positions":{"$ref":"relax#/positions_bohr"},"cell":{"$ref":"relax#/cell_bohr"},"coordinate_units":"bohr","charge":0.}}),
        task(
            "eos",
            "equation_of_state",
            json!({"volume_ratios":[0.94,0.97,1.,1.03,1.06]})
        )
    ]);
    let report = context()
        .run(|| execute_request(&request, Path::new(".")))
        .unwrap();
    assert_eq!(report["status"], "completed", "{report}");
    assert_eq!(result(&report, "relax")["converged"], true);
    assert!(
        (result(&report, "at_relaxed")["energy_hartree"]
            .as_f64()
            .unwrap()
            + 5.4464146416367)
            .abs()
            < 2e-10
    );
    assert_eq!(
        result(&report, "eos")["electronic_temperature_kelvin"],
        300.
    );
    assert!(
        result(&report, "eos")["fit"]["bulk_modulus_gpa"]
            .as_f64()
            .unwrap()
            > 0.
    );
}

#[test]
fn native_unit_postprocessors_cover_fit_moduli_thermo_and_qha() {
    let volumes: Vec<_> = [0.9, 0.95, 1., 1.05, 1.1]
        .into_iter()
        .map(|x| x * 100.)
        .collect();
    let energies: Vec<_> = volumes
        .iter()
        .map(|v| {
            let x = (100_f64 / v).powf(2. / 3.) - 1.;
            -2. + 9. * 100. * 0.01 / 16. * (4. * x.powi(3) + x * x * (2. - 4. * x))
        })
        .collect();
    let mut c = vec![vec![0.; 6]; 6];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = if i == j { 0.02 } else { 0.01 };
        }
    }
    for i in 3..6 {
        c[i][i] = 0.005;
    }
    let request = json!({"schema":"gfn1-rs-crystal-properties-v1","tasks":[
        task("fit","fit_equation_of_state",json!({"volumes_bohr3":volumes,"energies_hartree":energies})),
        task("moduli","elastic_properties",json!({"stiffness_hartree_per_bohr3":c,"density_kg_m3":3000.})),
        task("direction","directional_elastic_properties",json!({"stiffness_hartree_per_bohr3":c,"density_kg_m3":3000.,"direction":[1.,2.,3.],"transverse":[-2.,1.,0.]})),
        task("thermal","harmonic_thermodynamics",json!({"wavenumbers_cm1":[[0.,0.,0.,500.,600.,700.]],"weights":[1.],"temperatures_kelvin":[0.,300.]})),
        task("qha","quasiharmonic_zero_pressure",json!({"volumes_bohr3":volumes,"static_energies_hartree":energies,"phonons_by_volume_cm1":vec![vec![vec![500.,600.,700.]];5],"q_weights":[1.],"temperatures_kelvin":[0.,300.]}))]});
    let report = context()
        .run(|| execute_request(&request, Path::new(".")))
        .unwrap();
    assert_eq!(report["status"], "completed", "{report}");
    assert!(
        (result(&report, "fit")["equilibrium_volume_bohr3"]
            .as_f64()
            .unwrap()
            - 100.)
            .abs()
            < 1e-9
    );
    assert!(
        result(&report, "thermal")[1]["heat_capacity_hartree_per_kelvin"]
            .as_f64()
            .unwrap()
            > 0.
    );
    assert!(
        result(&report, "qha")[1]["thermal_expansion_per_kelvin"]
            .as_f64()
            .unwrap()
            .abs()
            < 1e-12
    );
    assert_eq!(operation_catalog().as_array().unwrap().len(), 21);
}

#[test]
fn errors_nonconvergence_and_bad_references_cannot_be_reported_as_success() {
    let mut request = request();
    request["continue_on_error"] = json!(true);
    request["tasks"] = json!([
        task(
            "bad",
            "harmonic_thermodynamics",
            json!({"wavenumbers_cm1":[[-100.,500.]],"weights":[1.],"temperatures_kelvin":[300.]})
        ),
        task(
            "dependent",
            "phonon_dos",
            json!({"wavenumbers_cm1":{"$ref":"bad#/wavenumbers_cm1"},"weights":[1.],"grid_cm1":[0.,100.]})
        ),
        task("after", "ground_state", json!({}))
    ]);
    let report = context()
        .run(|| execute_request(&request, Path::new(".")))
        .unwrap();
    assert_eq!(report["status"], "partial");
    assert_eq!(report["failed_tasks"], 2);
    assert_eq!(report["tasks"][0]["status"], "error");
    assert_eq!(report["tasks"][2]["status"], "completed");
    request["continue_on_error"] = json!(false);
    let report = context()
        .run(|| execute_request(&request, Path::new(".")))
        .unwrap();
    assert_eq!(report["skipped_tasks"], 2);
    request["tasks"] = json!([
        task(
            "cap",
            "relax_periodic",
            json!({"max_iterations":1,"force_tolerance_hartree_per_bohr":1e-14})
        ),
        task("skip", "ground_state", json!({}))
    ]);
    let report = context()
        .run(|| execute_request(&request, Path::new(".")))
        .unwrap();
    assert_eq!(report["tasks"][0]["status"], "not_converged");
    assert_eq!(report["status"], "partial");
    assert_eq!(report["skipped_tasks"], 1);
    request["electronic"]["temperature_k"] = json!(300.);
    assert!(context()
        .run(|| execute_request(&request, Path::new(".")))
        .unwrap_err()
        .to_string()
        .contains("unknown electronic field"));
}
