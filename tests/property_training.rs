// SPDX-License-Identifier: GPL-3.0-or-later
use gfn1_rs::ml::MlHamiltonian;
use gfn1_rs::property_training::{
    evaluate_property_sample, fit_properties, property_registry, PropertyDataset,
    PropertyGradientMethod, PropertyObservable as O, PropertySample, PropertySplit, PropertyTarget,
    PropertyTrainingOptions, TrainableParameter,
};
use gfn1_rs::{Gfn1Parameters, PeriodicSystem};
use serde_json::json;

fn model() -> MlHamiltonian {
    let mut m = MlHamiltonian::zero(vec![1], vec![1.3, 2.2], 5.);
    let h = m.heads.get_mut(&1).unwrap();
    h.levels[0].bias = 0.0012;
    h.levels[0].weights = vec![0.003, -0.002];
    h.correction.bias = 0.0003;
    h.correction.weights = vec![0.001, -0.0007];
    m.training_provenance = "nonzero user warm-start test model".into();
    m
}
fn sample(id: &str, split: PropertySplit, r: f64, with_hessian: bool) -> PropertySample {
    let system = PeriodicSystem::from_xyz_str(
        &format!("2\nsynthetic molecular ML property labels\nH 0 0 0\nH {r} 0 0\n"),
        0.,
        false,
    )
    .unwrap();
    let mut s = PropertySample::new(id, split, system);
    s.targets = vec![
        PropertyTarget::new(O::Energy, json!(0.), 0.1),
        PropertyTarget::new(O::Forces, json!([[0., 0., 0.], [0., 0., 0.]]), 0.1),
    ];
    if with_hessian {
        s.targets.push(PropertyTarget::new(
            O::Hessian,
            json!(vec![vec![0.; 6]; 6]),
            0.1,
        ));
    }
    s
}
fn data(samples: Vec<PropertySample>) -> PropertyDataset {
    PropertyDataset {
        samples,
        provenance: "Synthetic analytic teacher architecture test, not independent chemistry data"
            .into(),
        license_spdx: "GPL-3.0-or-later".into(),
        reference_method: "same native analytic ML Hamiltonian with a changed radial coordinate"
            .into(),
        independent_reference: false,
        source_sha256: None,
    }
}
fn label(samples: &mut [PropertySample], teacher: &MlHamiltonian, params: &Gfn1Parameters) {
    for s in samples {
        let prediction = evaluate_property_sample(teacher, params, s).unwrap();
        for t in &mut s.targets {
            t.values = prediction["predictions"][t.observable.name()].clone();
        }
    }
}

#[test]
fn analytic_property_fine_tuning_decreases_train_and_held_out_losses_and_reloads() {
    let params = Gfn1Parameters::builtin().unwrap();
    let mut student = model();
    let before = student.clone();
    let mut teacher = student.clone();
    teacher.heads.get_mut(&1).unwrap().correction.weights[0] = 0.009;
    let mut dataset = data(vec![
        sample("train-a", PropertySplit::Train, 0.67, true),
        sample("train-b", PropertySplit::Train, 0.92, true),
        sample("held-out", PropertySplit::Validation, 0.79, true),
    ]);
    label(&mut dataset.samples, &teacher, &params);
    let settings = PropertyTrainingOptions {
        parameters: vec![TrainableParameter::parse("H:correction:1").unwrap()],
        max_epochs: 50,
        learning_rate: 0.1,
        gradient_tolerance: 1e-10,
        loss_tolerance: 1e-18,
        ..Default::default()
    };
    let report = fit_properties(&mut student, &params, &dataset, &settings)
        .unwrap()
        .as_json();
    let initial = report["initial_train"]["loss"].as_f64().unwrap();
    let final_loss = report["final_train"]["loss"].as_f64().unwrap();
    eprintln!(
        "property fit: initial {initial:e}, final {final_loss:e}, termination {}",
        report["termination"]
    );
    assert!(initial > 1e-6 && final_loss < initial * 1e-5, "{report}");
    assert!(
        report["final_validation"]["loss"].as_f64().unwrap()
            < report["initial_validation"]["loss"].as_f64().unwrap() * 1e-5
    );
    assert!(report["gradient_method"]
        .as_str()
        .unwrap()
        .contains("central model-parameter"));
    assert_eq!(
        student.heads[&1].levels[0].weights,
        before.heads[&1].levels[0].weights
    );
    assert_eq!(
        student.heads[&1].correction.bias,
        before.heads[&1].correction.bias
    );
    assert_eq!(
        student.heads[&1].correction.weights[1],
        before.heads[&1].correction.weights[1]
    );
    assert_eq!(student.license_spdx, before.license_spdx);
    assert!(report["epochs"]
        .as_array()
        .unwrap()
        .iter()
        .all(|e| e["accepted"] == true));
    let path = std::path::PathBuf::from("target/validation/v070")
        .join(format!("property-trained-{}.model", std::process::id()));
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, student.to_model_string().unwrap()).unwrap();
    let (loaded, identity) = MlHamiltonian::resolve_with_identity(path.to_str().unwrap()).unwrap();
    assert_eq!(
        identity.canonical_sha256,
        report["final_model_canonical_sha256"]
    );
    assert_ne!(
        identity.canonical_sha256,
        report["initial_model_canonical_sha256"]
    );
    let a = evaluate_property_sample(&loaded, &params, &dataset.samples[2]).unwrap();
    let b = evaluate_property_sample(&student, &params, &dataset.samples[2]).unwrap();
    assert_eq!(a["predictions"], b["predictions"]);
    assert_eq!(a["conditions"]["electronic_temperature_kelvin"], 300.);
}

#[test]
fn existing_analytic_energy_force_weight_gradient_matches_parameter_difference() {
    let params = Gfn1Parameters::builtin().unwrap();
    let mut teacher = model();
    teacher.heads.get_mut(&1).unwrap().levels[0].weights[0] += 0.001;
    let mut dataset = data(vec![
        sample("a", PropertySplit::Train, 0.75, false),
        sample("b", PropertySplit::Validation, 0.82, false),
    ]);
    label(&mut dataset.samples, &teacher, &params);
    let mut a = model();
    let mut b = a.clone();
    let mut settings = PropertyTrainingOptions {
        parameters: vec![TrainableParameter::parse("1:s:1").unwrap()],
        max_epochs: 1,
        learning_rate: 1e-4,
        ..Default::default()
    };
    let analytic = fit_properties(&mut a, &params, &dataset, &settings)
        .unwrap()
        .as_json();
    settings.gradient_method = PropertyGradientMethod::CentralParameterDifference;
    let central = fit_properties(&mut b, &params, &dataset, &settings)
        .unwrap()
        .as_json();
    let ga = analytic["epochs"][0]["gradient_norm"].as_f64().unwrap();
    let gc = central["epochs"][0]["gradient_norm"].as_f64().unwrap();
    assert!(
        (ga - gc).abs() < 1e-7 * ga.abs().max(1.),
        "analytic {ga:e}, parameter difference {gc:e}"
    );
    assert!(analytic["gradient_method"]
        .as_str()
        .unwrap()
        .starts_with("analytic"));
    assert!((a.heads[&1].levels[0].weights[0] - b.heads[&1].levels[0].weights[0]).abs() < 1e-11);
}

#[test]
fn masks_units_splits_and_zero_sensitivity_are_explicit() {
    let params = Gfn1Parameters::builtin().unwrap();
    let mut student = model();
    let mut dataset = data(vec![
        sample("a", PropertySplit::Train, 0.75, false),
        sample("b", PropertySplit::Validation, 0.82, false),
    ]);
    label(&mut dataset.samples, &student, &params);
    for s in &mut dataset.samples {
        s.targets.retain(|t| t.observable == O::Forces);
        let t = &mut s.targets[0];
        let f = t.values[0][0].as_f64().unwrap();
        t.values[0][0] = json!(f + 0.1);
        t.values[0][1] = json!(1e6);
        t.mask = Some(json!([[true, false, false], [false, false, false]]));
        t.scale = 0.2;
        t.weight = 2.;
    }
    let original = student.heads[&1].correction.bias;
    let report = fit_properties(
        &mut student,
        &params,
        &dataset,
        &PropertyTrainingOptions {
            parameters: vec![TrainableParameter::parse("1:correction:0").unwrap()],
            max_epochs: 1,
            ..Default::default()
        },
    )
    .unwrap()
    .as_json();
    assert!((report["initial_train"]["loss"].as_f64().unwrap() - 0.25).abs() < 1e-12);
    assert_eq!(report["termination"], "zero_selected_parameter_sensitivity");
    assert_eq!(student.heads[&1].correction.bias, original);
    assert_eq!(report["insensitive_selected_parameters"], json!(["1:3:0"]));
    dataset.samples[1].split = PropertySplit::Train;
    assert!(dataset
        .validate()
        .unwrap_err()
        .to_string()
        .contains("held-out"));
    let mut json = json!({"schema":"gfn1-rs-property-training-v1","coordinate_units":"angstrom","provenance":"explicit source","license_spdx":"CC0-1.0","reference_method":"synthetic","independent_reference":false,
        "samples":[{"id":"a","split":"train","xyz":"2\nH2\nH 0 0 0\nH .75 0 0\n","charge":0,"electronic_temperature_kelvin":300,"targets":[{"observable":"energy","unit":"ev","values":-1,"scale":0.1}]}]});
    assert!(PropertyDataset::from_json(&json)
        .unwrap_err()
        .to_string()
        .contains("requires unit hartree"));
    json["samples"][0]["targets"][0]["unit"] = json!("hartree");
    assert!(PropertyDataset::from_json(&json)
        .unwrap_err()
        .to_string()
        .contains("held-out"));
    assert_eq!(property_registry().as_array().unwrap().len(), 17);
}

#[test]
fn nonzero_ml_periodic_registry_predicts_actual_bands_born_dielectric_and_analytic_elasticity() {
    let params = Gfn1Parameters::builtin().unwrap();
    let m = model();
    let system = PeriodicSystem::from_xyz_str(
        "2\nLattice=\"5 0 0 0 5 0 0 0 5\" pbc=\"T T T\"\nH 0 0 0\nH .75 .1 0\n",
        0.,
        false,
    )
    .unwrap();
    let mut s = PropertySample::new("periodic-nonzero-ml", PropertySplit::Train, system);
    s.periodic = Some(gfn1_rs::PbcOptions {
        ao_cutoff: 12.,
        ..Default::default()
    });
    s.band_kpoints = vec![[0.; 3], [0.25, 0., 0.]];
    s.masses_amu = vec![1.00794; 2];
    s.targets = vec![
        // GFN1 hydrogen has both 1s and 2s shells: four physical H2 bands.
        PropertyTarget::new(
            O::BandEnergies,
            json!([[0., 0., 0., 0.], [0., 0., 0., 0.]]),
            1.,
        ),
        PropertyTarget::new(
            O::BornEffectiveCharges,
            json!(vec![vec![vec![0.; 3]; 3]; 2]),
            1.,
        ),
        PropertyTarget::new(O::ElectronicDielectric, json!(vec![vec![0.; 3]; 3]), 1.),
        PropertyTarget::new(O::ElasticClamped, json!(vec![vec![0.; 6]; 6]), 1.),
    ];
    let value = evaluate_property_sample(&m, &params, &s).unwrap();
    assert_eq!(value["ml_model_sha256"], m.canonical_sha256().unwrap());
    assert_eq!(value["conditions"]["electronic_temperature_kelvin"], 300.);
    let bands = &value["predictions"]["band_energies"];
    assert!(bands[0][1].as_f64().unwrap() > bands[0][0].as_f64().unwrap());
    let eps = value["predictions"]["dielectric_electronic"][0][0]
        .as_f64()
        .unwrap();
    assert!(eps > 1.);
    let mut train = s.clone();
    train.targets.retain(|t| t.observable == O::BandEnergies);
    let mut validation = train.clone();
    validation.id = "held-out-periodic".into();
    validation.split = PropertySplit::Validation;
    let error = fit_properties(
        &mut m.clone(),
        &params,
        &data(vec![train, validation]),
        &PropertyTrainingOptions {
            parameters: vec![TrainableParameter::parse("1:correction:1").unwrap()],
            max_epochs: 1,
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("correction-only"));
    let mut unrelaxed = s.clone();
    unrelaxed.targets = vec![PropertyTarget::new(
        O::ElasticRelaxed,
        json!(vec![vec![0.; 6]; 6]),
        1.,
    )];
    let unavailable = evaluate_property_sample(&m, &params, &unrelaxed)
        .unwrap_err()
        .to_string();
    assert!(
        unavailable.contains("unavailable") && unavailable.contains("gradient"),
        "{unavailable}"
    );
    let mut bad = s.clone();
    bad.targets = vec![PropertyTarget::new(
        O::BandEnergies,
        json!([[0., 0., 0.]]),
        1.,
    )];
    assert!(evaluate_property_sample(&m, &params, &bad)
        .unwrap_err()
        .to_string()
        .contains("shape mismatch"));
}

#[test]
fn periodic_clamped_elastic_and_dielectric_objectives_fine_tune_one_level_weight() {
    fn flatten(value: &serde_json::Value) -> Vec<f64> {
        match value.as_array() {
            Some(array) => array.iter().flat_map(flatten).collect(),
            None => vec![value.as_f64().unwrap()],
        }
    }
    let params = Gfn1Parameters::builtin().unwrap();
    let mut student = model();
    let initial = student.clone();
    let mut teacher = student.clone();
    teacher.heads.get_mut(&1).unwrap().levels[0].weights[0] += 0.0007;
    teacher.training_provenance = "Synthetic nonzero periodic analytic-property teacher".into();
    let mut samples = Vec::new();
    for (id, split, distance) in [
        ("periodic-fit", PropertySplit::Train, 0.75),
        ("periodic-held-out", PropertySplit::Validation, 0.83),
    ] {
        let system = PeriodicSystem::from_xyz_str(
            &format!(
                "2\nLattice=\"5 0 0 0 5 0 0 0 5\" pbc=\"T T T\"\nH 0 0 0\nH {distance} .1 0\n"
            ),
            0.,
            false,
        )
        .unwrap();
        let mut sample = PropertySample::new(id, split, system);
        sample.periodic = Some(gfn1_rs::PbcOptions {
            ao_cutoff: 12.,
            ..Default::default()
        });
        sample.targets = vec![
            PropertyTarget::new(O::ElasticClamped, json!(vec![vec![0.; 6]; 6]), 1.),
            PropertyTarget::new(O::ElectronicDielectric, json!(vec![vec![0.; 3]; 3]), 1.),
        ];
        let reference = evaluate_property_sample(&teacher, &params, &sample).unwrap();
        let prediction = evaluate_property_sample(&student, &params, &sample).unwrap();
        for target in &mut sample.targets {
            let name = target.observable.name();
            target.values = reference["predictions"][name].clone();
            let a = flatten(&target.values);
            let b = flatten(&prediction["predictions"][name]);
            // A declared objective-specific scale normalizes the initial
            // teacher/student response difference. This is not label fitting
            // on held-out data: all optimizer gradients and line searches use
            // only the training split. Both raw tensor families must respond.
            target.scale = (a.iter().zip(&b).map(|(a, b)| (a - b).powi(2)).sum::<f64>()
                / a.len() as f64)
                .sqrt();
            assert!(
                target.scale > 1e-12,
                "{id}: {name} has no resolved level sensitivity"
            );
        }
        samples.push(sample);
    }
    // Use training-derived unit scales unchanged on the held-out structure.
    for t in 0..samples[0].targets.len() {
        samples[1].targets[t].scale = samples[0].targets[t].scale;
    }
    let mut dataset = data(samples);
    dataset.provenance = "Synthetic periodic clamped-C/electronic-dielectric targets; fixed cells and 300 K; no independent accuracy claim".into();
    let settings = PropertyTrainingOptions {
        parameters: vec![TrainableParameter::parse("1:s:1").unwrap()],
        max_epochs: 12,
        learning_rate: 0.05,
        parameter_difference_step: 2e-6,
        maximum_parameter_step: 2e-4,
        gradient_tolerance: 1e-8,
        loss_tolerance: 1e-12,
        gradient_method: PropertyGradientMethod::CentralParameterDifference,
        ..Default::default()
    };
    let report = fit_properties(&mut student, &params, &dataset, &settings)
        .unwrap()
        .as_json();
    for split in ["train", "validation"] {
        let initial_loss = report[format!("initial_{split}")]["loss"].as_f64().unwrap();
        let final_loss = report[format!("final_{split}")]["loss"].as_f64().unwrap();
        eprintln!("periodic property fit {split}: {initial_loss:e} -> {final_loss:e}");
        assert!(final_loss < initial_loss * 1e-3, "{report}");
        for property in ["elastic_clamped", "dielectric_electronic"] {
            let before = report[format!("initial_{split}")]["by_property"][property]["rmse_native"]
                .as_f64()
                .unwrap();
            let after = report[format!("final_{split}")]["by_property"][property]["rmse_native"]
                .as_f64()
                .unwrap();
            assert!(
                after < before * 0.04,
                "{split}/{property}: {before:e} -> {after:e}"
            );
        }
    }
    assert!(report["gradient_method"]
        .as_str()
        .unwrap()
        .starts_with("central model-parameter"));
    assert_eq!(
        student.heads[&1].levels[0].bias,
        initial.heads[&1].levels[0].bias
    );
    assert_eq!(
        student.heads[&1].levels[0].weights[1],
        initial.heads[&1].levels[0].weights[1]
    );
    assert_eq!(
        student.heads[&1].correction.weights,
        initial.heads[&1].correction.weights
    );
    assert_eq!(
        student.heads[&1].correction.bias,
        initial.heads[&1].correction.bias
    );
    let path = std::path::PathBuf::from("target/validation/v070").join(format!(
        "periodic-property-trained-{}.model",
        std::process::id()
    ));
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, student.to_model_string().unwrap()).unwrap();
    let (loaded, identity) = MlHamiltonian::resolve_with_identity(path.to_str().unwrap()).unwrap();
    assert_eq!(
        identity.canonical_sha256,
        report["final_model_canonical_sha256"]
    );
    for sample in &dataset.samples {
        let original = evaluate_property_sample(&student, &params, sample).unwrap();
        let reloaded = evaluate_property_sample(&loaded, &params, sample).unwrap();
        assert_eq!(original["predictions"], reloaded["predictions"]);
        assert_eq!(
            reloaded["conditions"]["electronic_temperature_kelvin"],
            300.
        );
        assert_eq!(reloaded["conditions"]["kmesh"], json!([1, 1, 1]));
    }
    std::fs::write(
        "target/validation/v070/periodic-property-training-report.json",
        serde_json::to_string_pretty(&report).unwrap() + "\n",
    )
    .unwrap();
}
