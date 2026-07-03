// SPDX-License-Identifier: GPL-3.0-or-later

use gfn1_rs::{
    analytic_gradient, run_electronic, AnalyticGradientOptions, ElectronicOptions, Gfn1Parameters,
    PeriodicSystem,
};

struct TestMolecule {
    name: &'static str,
    xyz: &'static str,
    gradient_fd: bool,
}

const TEST_MOLECULES: &[TestMolecule] = &[
    TestMolecule {
        name: "hydrogen",
        xyz: "2\nhydrogen\nH 0.000000 0.000000 0.000000\nH 0.740000 0.000000 0.000000\n",
        gradient_fd: true,
    },
    TestMolecule {
        name: "hydrogen fluoride",
        xyz: "2\nhydrogen fluoride\nH 0.000000 0.000000 0.000000\nF 0.917000 0.000000 0.000000\n",
        gradient_fd: true,
    },
    TestMolecule {
        name: "water",
        xyz: "3\nwater\nO 0.000000 0.000000 0.000000\nH 0.757000 0.586000 0.000000\nH -0.757000 0.586000 0.000000\n",
        gradient_fd: true,
    },
    TestMolecule {
        name: "phosphine",
        xyz: "4\nphosphine\nP 0.000000 0.000000 0.000000\nH 1.193000 0.000000 0.768000\nH -0.596500 1.033300 0.768000\nH -0.596500 -1.033300 0.768000\n",
        gradient_fd: true,
    },
    TestMolecule {
        name: "ferrocene",
        xyz: "21\nferrocene staggered test geometry\nFe 0.000000 0.000000 0.000000\nC 1.430000 0.000000 1.650000\nC 0.441908 1.360370 1.650000\nC -1.156908 0.840788 1.650000\nC -1.156908 -0.840788 1.650000\nC 0.441908 -1.360370 1.650000\nH 2.510000 0.000000 1.650000\nH 0.775615 2.386978 1.650000\nH -2.030615 1.475161 1.650000\nH -2.030615 -1.475161 1.650000\nH 0.775615 -2.386978 1.650000\nC 1.156908 0.840788 -1.650000\nC -0.441908 1.360370 -1.650000\nC -1.430000 0.000000 -1.650000\nC -0.441908 -1.360370 -1.650000\nC 1.156908 -0.840788 -1.650000\nH 2.030615 1.475161 -1.650000\nH -0.775615 2.386978 -1.650000\nH -2.510000 0.000000 -1.650000\nH -0.775615 -2.386978 -1.650000\nH 2.030615 -1.475161 -1.650000\n",
        gradient_fd: false,
    },
    TestMolecule {
        name: "borane",
        xyz: "4\nborane\nB 0.000000 0.000000 0.000000\nH 1.190000 0.000000 0.000000\nH -0.595000 1.030570 0.000000\nH -0.595000 -1.030570 0.000000\n",
        gradient_fd: true,
    },
    TestMolecule {
        name: "caffeine",
        xyz: "24\ncaffeine fixed test geometry\nN 0.000000 0.000000 0.000000\nC 1.250000 0.000000 0.000000\nN 2.000000 1.100000 0.000000\nC 1.250000 2.200000 0.000000\nC 0.000000 2.200000 0.000000\nC -0.700000 1.100000 0.000000\nN 1.750000 3.350000 0.000000\nC 0.750000 4.250000 0.000000\nN -0.350000 3.350000 0.000000\nO 1.900000 -1.050000 0.000000\nO -1.950000 1.100000 0.000000\nC -0.800000 -1.200000 0.250000\nH -1.830000 -0.880000 0.250000\nH -0.550000 -1.780000 1.140000\nH -0.550000 -1.820000 -0.620000\nC 3.450000 1.100000 0.250000\nH 3.800000 2.130000 0.250000\nH 3.780000 0.580000 1.150000\nH 3.850000 0.540000 -0.600000\nC 3.100000 3.900000 0.250000\nH 3.060000 4.990000 0.250000\nH 3.640000 3.580000 1.140000\nH 3.700000 3.520000 -0.580000\nH 0.780000 5.330000 0.000000\n",
        gradient_fd: false,
    },
];

#[test]
fn h2_nonpbc_singlepoint_with_external_param() {
    let Ok(param_path) = std::env::var("GFN1_XTB_PARAM") else {
        return;
    };
    let params = Gfn1Parameters::from_file(param_path).unwrap();
    let system =
        PeriodicSystem::from_xyz_str("2\nH2\nH 0.0 0.0 0.0\nH 0.74 0.0 0.0\n", 0.0, false).unwrap();
    let result = run_electronic(&system, &params, ElectronicOptions::default()).unwrap();
    assert!(result.converged);
    assert!(result.total_free.is_finite());
    assert!(result.repulsion_energy > 0.0);
    assert!(result.iterations > 0);
}

#[test]
fn charged_doublet_h2_accepts_charge_and_spin_multiplicity() {
    let Ok(param_path) = std::env::var("GFN1_XTB_PARAM") else {
        return;
    };
    let params = Gfn1Parameters::from_file(param_path).unwrap();
    let system =
        PeriodicSystem::from_xyz_str("2\nH2+\nH 0.0 0.0 0.0\nH 0.74 0.0 0.0\n", 0.0, false)
            .unwrap();
    let mut options = ElectronicOptions::default();
    options.charge = Some(1.0);
    options.spin_multiplicity = Some(2);
    options.electronic_temperature = 0.0;
    options.enable_dispersion = false;
    let result = run_electronic(&system, &params, options).unwrap();
    assert!(result.converged);
    assert!((result.nelec - 1.0).abs() < 1.0e-12);
    assert!((result.occupations.iter().sum::<f64>() - 1.0).abs() < 1.0e-10);
    assert!(
        result
            .occupations
            .iter()
            .any(|occ| (occ - 1.0).abs() < 1.0e-10),
        "doublet H2+ should contain a singly occupied orbital: {:?}",
        result.occupations
    );
    let qsum: f64 = result.atomic_charges.iter().sum();
    assert!((qsum - 1.0).abs() < 1.0e-6, "atomic charge sum {qsum}");
}

#[test]
fn requested_molecules_nonpbc_singlepoint_with_external_param() {
    let Ok(param_path) = std::env::var("GFN1_XTB_PARAM") else {
        return;
    };
    let params = Gfn1Parameters::from_file(param_path).unwrap();
    for molecule in TEST_MOLECULES {
        let system = PeriodicSystem::from_xyz_str(molecule.xyz, 0.0, false).unwrap();
        let options = ElectronicOptions::default();
        let result = run_electronic(&system, &params, options)
            .unwrap_or_else(|err| panic!("{} singlepoint failed: {err}", molecule.name));
        assert!(result.converged, "{} did not converge", molecule.name);
        assert!(
            result.total_free.is_finite(),
            "{} total energy is not finite",
            molecule.name
        );
    }
}

#[test]
fn requested_small_molecule_analytic_gradients_match_finite_difference_for_current_terms() {
    let Ok(param_path) = std::env::var("GFN1_XTB_PARAM") else {
        return;
    };
    let params = Gfn1Parameters::from_file(param_path).unwrap();
    for molecule in TEST_MOLECULES.iter().filter(|m| m.gradient_fd) {
        let system = PeriodicSystem::from_xyz_str(molecule.xyz, 0.0, false).unwrap();
        let grad_options = AnalyticGradientOptions::default();
        let gradient = analytic_gradient(&system, &params, grad_options.clone())
            .unwrap_or_else(|err| panic!("{} analytic gradient failed: {err}", molecule.name));

        let h = 1.0e-4;
        for atom in 0..system.atoms.len() {
            for component in 0..3 {
                let mut plus = system.clone();
                let mut minus = system.clone();
                shift_component(&mut plus, atom, component, h);
                shift_component(&mut minus, atom, component, -h);
                let ep = run_electronic(&plus, &params, grad_options.electronic.clone())
                    .unwrap_or_else(|err| {
                        panic!(
                            "{} +FD atom {atom} component {component} failed: {err}",
                            molecule.name
                        )
                    })
                    .total_free;
                let em = run_electronic(&minus, &params, grad_options.electronic.clone())
                    .unwrap_or_else(|err| {
                        panic!(
                            "{} -FD atom {atom} component {component} failed: {err}",
                            molecule.name
                        )
                    })
                    .total_free;
                let fd = (ep - em) / (2.0 * h);
                let an = component_value(gradient.gradient[atom], component);
                assert!(
                    (an - fd).abs() < 5.0e-5,
                    "{} atom {atom} component {component}: analytic {an} finite-diff {fd}",
                    molecule.name
                );
            }
        }
    }
}

#[test]
fn requested_large_molecule_single_coordinate_gradient_probe() {
    let Ok(param_path) = std::env::var("GFN1_XTB_PARAM") else {
        return;
    };
    let params = Gfn1Parameters::from_file(param_path).unwrap();
    for molecule in TEST_MOLECULES.iter().filter(|m| !m.gradient_fd) {
        let system = PeriodicSystem::from_xyz_str(molecule.xyz, 0.0, false).unwrap();
        let grad_options = AnalyticGradientOptions::default();
        let gradient = analytic_gradient(&system, &params, grad_options.clone())
            .unwrap_or_else(|err| panic!("{} analytic gradient failed: {err}", molecule.name));

        let h = 1.0e-4;
        let mut plus = system.clone();
        let mut minus = system.clone();
        shift_component(&mut plus, 0, 0, h);
        shift_component(&mut minus, 0, 0, -h);
        let ep = run_electronic(&plus, &params, grad_options.electronic.clone())
            .unwrap()
            .total_free;
        let em = run_electronic(&minus, &params, grad_options.electronic.clone())
            .unwrap()
            .total_free;
        let fd = (ep - em) / (2.0 * h);
        let an = gradient.gradient[0].x;
        assert!(
            (an - fd).abs() < 1.0e-4,
            "{} atom 0 x: analytic {an} finite-diff {fd}",
            molecule.name
        );
    }
}

fn shift_component(system: &mut PeriodicSystem, atom: usize, component: usize, delta: f64) {
    match component {
        0 => system.atoms[atom].position.x += delta,
        1 => system.atoms[atom].position.y += delta,
        _ => system.atoms[atom].position.z += delta,
    }
}

fn component_value(value: gfn1_rs::math::Vec3, component: usize) -> f64 {
    match component {
        0 => value.x,
        1 => value.y,
        _ => value.z,
    }
}
