use gfn1_rs::dispersion::{
    d4_dispersion_energy, d4_dispersion_energy_gradient, D4DispersionOptions, D4_GFN2_DEFAULT_S9,
};
use gfn1_rs::{
    analytic_gradient, run_electronic, AnalyticGradientOptions, ElectronicOptions, Gfn1Parameters,
    PeriodicSystem,
};

const ASYMMETRIC_WATER: &str = "3
asymmetric water
O 0.000000 0.100000 -0.020000
H 0.820000 0.610000 0.030000
H -0.750000 0.680000 -0.040000
";

fn load_params() -> Option<Gfn1Parameters> {
    let path = std::env::var("GFN1_XTB_PARAM").ok()?;
    Some(Gfn1Parameters::from_file(path).unwrap())
}

fn component(v: &gfn1_rs::math::Vec3, axis: usize) -> f64 {
    match axis {
        0 => v.x,
        1 => v.y,
        2 => v.z,
        _ => panic!("invalid axis"),
    }
}

fn displaced(system: &PeriodicSystem, atom: usize, axis: usize, delta_bohr: f64) -> PeriodicSystem {
    let mut out = system.clone();
    match axis {
        0 => out.atoms[atom].position.x += delta_bohr,
        1 => out.atoms[atom].position.y += delta_bohr,
        2 => out.atoms[atom].position.z += delta_bohr,
        _ => panic!("invalid axis"),
    }
    out
}

#[test]
fn d4_s9_default_depends_on_d4_activation() {
    let stock = ElectronicOptions::default();
    assert_eq!(stock.d4_dispersion_options().s9, 0.0);

    let d4 = ElectronicOptions {
        experimental_d4: true,
        ..ElectronicOptions::default()
    };
    assert_eq!(d4.d4_dispersion_options().s9, D4_GFN2_DEFAULT_S9);

    let explicit_zero = ElectronicOptions {
        experimental_d4: true,
        d4_s9: Some(0.0),
        ..ElectronicOptions::default()
    };
    assert_eq!(explicit_zero.d4_dispersion_options().s9, 0.0);
}

#[test]
fn fixed_charge_d4_gradient_matches_finite_difference() {
    let Some(params) = load_params() else {
        return;
    };
    let system = PeriodicSystem::from_xyz_str(ASYMMETRIC_WATER, 0.0, false).unwrap();
    let charges = vec![-0.18, 0.08, 0.10];
    let options = D4DispersionOptions::default();
    let analytic = d4_dispersion_energy_gradient(&system, &params, &charges, options).unwrap();
    let h = 1.0e-4;
    let probes = [(0, 0), (1, 1), (2, 2)];
    let mut max_delta: f64 = 0.0;
    for (atom, axis) in probes {
        let plus = displaced(&system, atom, axis, h);
        let minus = displaced(&system, atom, axis, -h);
        let fd = (d4_dispersion_energy(&plus, &params, &charges, options).unwrap()
            - d4_dispersion_energy(&minus, &params, &charges, options).unwrap())
            / (2.0 * h);
        let an = component(&analytic.gradient[atom], axis);
        max_delta = max_delta.max((an - fd).abs());
    }
    assert!(
        max_delta < 2.0e-7,
        "fixed-charge D4 gradient finite-difference max delta {max_delta:.3e}"
    );
}

#[test]
fn self_consistent_d4_gradient_matches_total_energy_finite_difference() {
    let Some(params) = load_params() else {
        return;
    };
    let system = PeriodicSystem::from_xyz_str(ASYMMETRIC_WATER, 0.0, false).unwrap();
    let electronic = ElectronicOptions {
        experimental_d4: true,
        energy_tolerance: 1.0e-10,
        charge_tolerance: 1.0e-8,
        max_scc: 1000,
        ..ElectronicOptions::default()
    };
    let grad_options = AnalyticGradientOptions {
        electronic: electronic.clone(),
        ..AnalyticGradientOptions::default()
    };
    let analytic = analytic_gradient(&system, &params, grad_options).unwrap();
    assert!(analytic.electronic_result.converged);
    assert!(analytic.electronic_result.dispersion_energy.is_finite());

    let h = 1.0e-4;
    let probes = [(0, 0), (1, 1), (2, 2)];
    let mut max_delta: f64 = 0.0;
    for (atom, axis) in probes {
        let plus = displaced(&system, atom, axis, h);
        let minus = displaced(&system, atom, axis, -h);
        let ep = run_electronic(&plus, &params, electronic.clone())
            .unwrap()
            .total_free;
        let em = run_electronic(&minus, &params, electronic.clone())
            .unwrap()
            .total_free;
        let fd = (ep - em) / (2.0 * h);
        let an = component(&analytic.gradient[atom], axis);
        max_delta = max_delta.max((an - fd).abs());
    }
    assert!(
        max_delta < 2.0e-5,
        "self-consistent D4 total gradient finite-difference max delta {max_delta:.3e}"
    );
}
