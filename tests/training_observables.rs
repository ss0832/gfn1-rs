// SPDX-License-Identifier: GPL-3.0-or-later
use gfn1_rs::math::Vec3;
use gfn1_rs::ml::*;
use gfn1_rs::training::*;
use gfn1_rs::*;
use std::{collections::BTreeMap, sync::Arc};
fn fixture() -> (
    MlHamiltonian,
    Gfn1Parameters,
    ElectronicOptions,
    Vec<TrainingExample>,
) {
    let system = PeriodicSystem::from_xyz_str(
        "3\nasymmetric water\nO .1 .2 0\nH .95 .73 .03\nH -.64 .86 -.04\n",
        0.0,
        false,
    )
    .unwrap();
    let mut model = MlHamiltonian::zero(vec![1, 8], vec![1.5, 3.0], 7.0);
    for (&z, h) in &mut model.heads {
        for (l, p) in h.levels.iter_mut().enumerate() {
            p.bias = 0.0001 * (l + 1) as f64;
            for (k, w) in p.weights.iter_mut().enumerate() {
                *w = 0.0002 * ((k + z as usize + l) % 3) as f64;
            }
        }
        h.correction.weights[0] = 0.0001;
    }
    let params = Gfn1Parameters::builtin().unwrap();
    let options = ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1e-12,
        charge_tolerance: 1e-11,
        ..Default::default()
    };
    let mut o = options.clone();
    o.ml_model = Some(Arc::new(model.clone()));
    let actual = analytic_gradient(
        &system,
        &params,
        AnalyticGradientOptions {
            electronic: o,
            ..Default::default()
        },
    )
    .unwrap();
    let mut sample = TrainingExample::new(system);
    sample.energy_hartree = Some(actual.total_energy - 0.003);
    sample.gradient_hartree_per_bohr = Some(
        actual
            .gradient
            .iter()
            .enumerate()
            .map(|(i, &g)| g + Vec3::new(0.0003 * (i + 1) as f64, -0.0002, 0.0001))
            .collect(),
    );
    sample.dipole_e_bohr = Some(actual.electronic_result.dipole + Vec3::new(0.03, -0.02, 0.01));
    (model, params, options, vec![sample])
}
#[test]
fn energy_gradient_and_dipole_parameter_loss_gradients_match_finite_difference() {
    let (model, params, options, data) = fixture();
    for (e, g, m) in [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.3, 0.2),
    ] {
        let config = TrainingConfig {
            energy_weight: e,
            gradient_weight: g,
            dipole_weight: m,
            ..Default::default()
        };
        let actual = loss_gradient(&model, &params, &options, &data, &[], &config).unwrap();
        for (z, l, index) in [(1, 0, Some(2)), (8, 1, None), (1, 3, Some(0))] {
            let mut plus = model.clone();
            let mut minus = model.clone();
            let step = 2e-6;
            let update = |model: &mut MlHamiltonian, value: f64| {
                let heads = model.heads.get_mut(&z).unwrap();
                let h = if l == 3 {
                    &mut heads.correction
                } else {
                    &mut heads.levels[l]
                };
                if let Some(k) = index {
                    h.weights[k] += value;
                } else {
                    h.bias += value;
                }
            };
            update(&mut plus, step);
            update(&mut minus, -step);
            let fd = (loss_gradient(&plus, &params, &options, &data, &[], &config)
                .unwrap()
                .loss
                .total
                - loss_gradient(&minus, &params, &options, &data, &[], &config)
                    .unwrap()
                    .loss
                    .total)
                / (2.0 * step);
            let heads = &actual.gradient[&z];
            let h = if l == 3 {
                &heads.correction
            } else {
                &heads.levels[l]
            };
            let predicted = index.map_or(h.bias, |i| h.weights[i]);
            assert!(
                (fd - predicted).abs() < 2e-7,
                "weights E={e} G={g} M={m}, Z={z} l={l}: implicit {predicted:e}, FD {fd:e}"
            );
        }
    }
}
#[test]
fn wtmad2_uses_reference_scales_case_counts_and_manual_subset_ratios() {
    let relative = vec![
        RelativeEnergyTarget {
            group: "small".into(),
            terms: vec![(0, 1.0)],
            reference_hartree: 10.0 / HARTREE_TO_KCAL_MOL,
        },
        RelativeEnergyTarget {
            group: "small".into(),
            terms: vec![(1, 1.0)],
            reference_hartree: 10.0 / HARTREE_TO_KCAL_MOL,
        },
        RelativeEnergyTarget {
            group: "large".into(),
            terms: vec![(2, 1.0)],
            reference_hartree: 100.0 / HARTREE_TO_KCAL_MOL,
        },
    ];
    let energies: Vec<f64> = [9.0, 11.0, 98.0]
        .iter()
        .map(|x| x / HARTREE_TO_KCAL_MOL)
        .collect();
    let config = TrainingConfig::default();
    let expected = (2.0 * 56.84 / 10.0 + 56.84 / 100.0 * 2.0) / 3.0;
    assert!((wtmad2(&energies, &relative, &config).unwrap() - expected).abs() < 1e-12);
    let mut config = TrainingConfig {
        groups: BTreeMap::from([
            (
                "small".into(),
                GroupWeight {
                    ratio: 0.25,
                    ..Default::default()
                },
            ),
            (
                "large".into(),
                GroupWeight {
                    ratio: 2.0,
                    ..Default::default()
                },
            ),
        ]),
        ..Default::default()
    };
    assert!((wtmad2(&energies, &relative, &config).unwrap() - 2.04624).abs() < 1e-12);
    config.groups.get_mut("small").unwrap().ratio = 0.0;
    assert!((wtmad2(&energies, &relative, &config).unwrap() - 1.1368).abs() < 1e-12);
}
#[test]
fn relative_energy_wtmad2_training_gradient_includes_all_reaction_partners() {
    let (model, params, options, mut data) = fixture();
    let mut second = data[0].clone();
    second.system.atoms[1].position.x += 0.2;
    for sample in &mut data {
        sample.energy_hartree = None;
        sample.gradient_hartree_per_bohr = None;
        sample.dipole_e_bohr = None;
    }
    second.energy_hartree = None;
    second.gradient_hartree_per_bohr = None;
    second.dipole_e_bohr = None;
    data.push(second);
    let relative = vec![RelativeEnergyTarget {
        group: "reaction".into(),
        terms: vec![(0, -1.0), (1, 1.0)],
        reference_hartree: 0.01,
    }];
    let config = TrainingConfig {
        energy_weight: 0.0,
        gradient_weight: 0.0,
        dipole_weight: 0.0,
        energy_weighting: EnergyWeighting::Wtmad2,
        energy_loss: EnergyLoss::Absolute,
        ..Default::default()
    };
    let result = loss_gradient(&model, &params, &options, &data, &relative, &config).unwrap();
    let step = 2e-6;
    let mut plus = model.clone();
    let mut minus = model;
    plus.heads.get_mut(&1).unwrap().levels[0].weights[2] += step;
    minus.heads.get_mut(&1).unwrap().levels[0].weights[2] -= step;
    let fd = (loss_gradient(&plus, &params, &options, &data, &relative, &config)
        .unwrap()
        .loss
        .total
        - loss_gradient(&minus, &params, &options, &data, &relative, &config)
            .unwrap()
            .loss
            .total)
        / (2.0 * step);
    assert!((fd - result.gradient[&1].levels[0].weights[2]).abs() < 1e-6);
}
