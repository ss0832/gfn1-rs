// SPDX-License-Identifier: GPL-3.0-or-later
//! Scientific acceptance against independently computed held-out structures.
use gfn1_rs::math::Vec3;
use gfn1_rs::ml::MlHamiltonian;
use gfn1_rs::*;
use serde_json::Value;
use std::{collections::BTreeMap, sync::Arc};

fn options(model: bool) -> ElectronicOptions {
    ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1e-12,
        charge_tolerance: 1e-10,
        ml_model: model
            .then(|| Arc::new(MlHamiltonian::resolve("builtin:water-clusters").unwrap())),
        ..Default::default()
    }
}
fn system(sample: &Value) -> PeriodicSystem {
    PeriodicSystem::from_xyz_str(sample["xyz"].as_str().unwrap(), 0.0, false).unwrap()
}
#[test]
fn tiny_builtin_improves_held_out_cluster_energy_gradient_and_dipole() {
    assert!(include_bytes!("../models/water-clusters-trained.model").len() < 5_000_000);
    let data: Value =
        serde_json::from_str(include_str!("../models/water-clusters-reference.json")).unwrap();
    let params = Gfn1Parameters::builtin().unwrap();
    let samples = data["samples"].as_array().unwrap();
    let anchor = &samples[0];
    assert_eq!(anchor["split"], "reference");
    assert!(!samples
        .iter()
        .any(|s| s["water_molecules"] == 4 && s["split"] == "train"));
    let mut errors = [[0.; 3]; 2];
    let mut counts = [0usize; 3];
    for (prediction, model) in [false, true].into_iter().enumerate() {
        let anchor_energy = run_electronic(&system(anchor), &params, options(model))
            .unwrap()
            .total_free;
        let mut energies = BTreeMap::new();
        for sample in samples.iter().filter(|s| s["split"] == "validation") {
            let result = analytic_gradient(
                &system(sample),
                &params,
                AnalyticGradientOptions {
                    electronic: options(model),
                    ..Default::default()
                },
            )
            .unwrap();
            energies.insert(sample["id"].as_str().unwrap(), result.total_energy);
            for (gradient, target) in result
                .gradient
                .iter()
                .zip(sample["gradient_hartree_per_bohr"].as_array().unwrap())
            {
                let v = Vec3::new(
                    target[0].as_f64().unwrap(),
                    target[1].as_f64().unwrap(),
                    target[2].as_f64().unwrap(),
                );
                errors[prediction][1] += (*gradient - v).norm2();
                if !model {
                    counts[1] += 3;
                }
            }
            let target = &sample["dipole_e_bohr"];
            let dipole = Vec3::new(
                target[0].as_f64().unwrap(),
                target[1].as_f64().unwrap(),
                target[2].as_f64().unwrap(),
            );
            errors[prediction][2] += (result.electronic_result.dipole - dipole).norm2();
            if !model {
                counts[2] += 3;
            }
        }
        for case in data["relative_energies"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| r["split"] == "validation")
        {
            let target = case["energy_hartree"].as_f64().unwrap();
            let mut energy = 0.;
            for term in case["terms"].as_array().unwrap() {
                let id = term["sample"].as_str().unwrap();
                energy += term["coefficient"].as_f64().unwrap()
                    * if id == "water-anchor" {
                        anchor_energy
                    } else {
                        energies[id]
                    };
            }
            errors[prediction][0] += (energy - target).powi(2);
            if !model {
                counts[0] += 1;
            }
        }
    }
    assert_eq!(counts, [8, 180, 24]);
    for k in 0..3 {
        let baseline = (errors[0][k] / counts[k] as f64).sqrt();
        let model = (errors[1][k] / counts[k] as f64).sqrt();
        assert!(
            model < 0.7 * baseline,
            "observable {k}: baseline {baseline:e}, trained {model:e}"
        );
    }
    let citations = gfn1_rs::citations::electronic_citations(&options(true), false);
    assert!(citations.contains(&"Adamo1999PBE0"));
    assert!(gfn1_rs::citations::bibliography(&citations)
        .unwrap()
        .contains("10.1039/B508541A"));
}
#[test]
fn trained_model_force_matches_energy_derivative_on_unseen_tetramer() {
    let data: Value =
        serde_json::from_str(include_str!("../models/water-clusters-reference.json")).unwrap();
    let sample = data["samples"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["water_molecules"] == 4 && s["split"] == "validation")
        .unwrap();
    let original = system(sample);
    let params = Gfn1Parameters::builtin().unwrap();
    let gradient = analytic_gradient(
        &original,
        &params,
        AnalyticGradientOptions {
            electronic: options(true),
            ..Default::default()
        },
    )
    .unwrap();
    let direction = Vec3::new(0.3, -0.2, 0.4);
    for h in [2e-4, 1e-4] {
        let mut plus = original.clone();
        let mut minus = original.clone();
        plus.atoms[1].position += direction * h;
        minus.atoms[1].position -= direction * h;
        let difference = (run_electronic(&plus, &params, options(true))
            .unwrap()
            .total_free
            - run_electronic(&minus, &params, options(true))
                .unwrap()
                .total_free)
            / (2. * h);
        assert!((difference - gradient.gradient[1].dot(direction)).abs() < 3e-7);
    }
}
