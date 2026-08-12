use gfn1_rs::basis::BasisOptions;
use gfn1_rs::coulomb::{
    coulomb_energy_potential_from_matrix, effective_coulomb_matrix, ShellChargeModel,
};
use gfn1_rs::electronic::{electronic_energy, mulliken_shell_charges};
use gfn1_rs::hamiltonian::build_h0;
use gfn1_rs::{
    analytic_gradient, run_electronic, AnalyticGradientOptions, ElectronicOptions, Gfn1Parameters,
    PeriodicSystem,
};

const CAFFEINE_XYZ: &str = "24
caffeine fixed test geometry
N 0.000000 0.000000 0.000000
C 1.250000 0.000000 0.000000
N 2.000000 1.100000 0.000000
C 1.250000 2.200000 0.000000
C 0.000000 2.200000 0.000000
C -0.700000 1.100000 0.000000
N 1.750000 3.350000 0.000000
C 0.750000 4.250000 0.000000
N -0.350000 3.350000 0.000000
O 1.900000 -1.050000 0.000000
O -1.950000 1.100000 0.000000
C -0.800000 -1.200000 0.250000
H -1.830000 -0.880000 0.250000
H -0.550000 -1.780000 1.140000
H -0.550000 -1.820000 -0.620000
C 3.450000 1.100000 0.250000
H 3.800000 2.130000 0.250000
H 3.780000 0.580000 1.150000
H 3.850000 0.540000 -0.600000
C 3.100000 3.900000 0.250000
H 3.060000 4.990000 0.250000
H 3.640000 3.580000 1.140000
H 3.700000 3.520000 -0.580000
H 0.780000 5.330000 0.000000
";

#[test]
fn caffeine_gradient_fd_probe() {
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    let system = PeriodicSystem::from_xyz_str(CAFFEINE_XYZ, 0.0, false).unwrap();
    let mut options = AnalyticGradientOptions::default();
    let tight = std::env::var_os("GFN1_FD_TIGHT").is_some();
    if tight {
        options.electronic.energy_tolerance = 1.0e-10;
        options.electronic.charge_tolerance = 1.0e-8;
        options.electronic.max_scc = 1000;
    }
    if let Ok(path) = std::env::var("GFN1_D3_REFERENCE") {
        options.electronic.d3_reference_path = Some(path);
    }
    let analytic = analytic_gradient(&system, &params, options.clone()).unwrap();
    let mut no_scc_grad_options = options.clone();
    no_scc_grad_options.include_scc = false;
    let no_scc_analytic = analytic_gradient(&system, &params, no_scc_grad_options.clone()).unwrap();
    let mut no_cn_options = options.clone();
    no_cn_options.electronic.hamiltonian.enable_cn_hamiltonian = false;
    let no_cn_analytic = analytic_gradient(&system, &params, no_cn_options.clone()).unwrap();
    let mut no_cn_no_scc_options = no_cn_options.clone();
    no_cn_no_scc_options.include_scc = false;
    let no_cn_no_scc_analytic =
        analytic_gradient(&system, &params, no_cn_no_scc_options.clone()).unwrap();
    let h = 1.0e-4;
    let probes: &[(usize, usize)] = if tight {
        &[(0, 1), (1, 2), (5, 0)]
    } else {
        &[
            (0usize, 1usize),
            (0, 2),
            (1, 1),
            (1, 2),
            (2, 2),
            (3, 2),
            (6, 1),
            (8, 1),
        ]
    };
    for &(atom, axis) in probes {
        let fd_total =
            finite_difference(&system, &params, &options.electronic, atom, axis, h, |e| {
                e.total_free
            });
        let fd_elec =
            finite_difference(&system, &params, &options.electronic, atom, axis, h, |e| {
                e.electronic_energy + e.isotropic_scc_energy + e.third_order_energy
            });
        let fd_rep = finite_difference(&system, &params, &options.electronic, atom, axis, h, |e| {
            e.repulsion_energy
        });
        let fd_disp =
            finite_difference(&system, &params, &options.electronic, atom, axis, h, |e| {
                e.dispersion_energy
            });
        let fd_hal = finite_difference(&system, &params, &options.electronic, atom, axis, h, |e| {
            e.halogen_energy
        });
        let (matrix_h0, matrix_scc, matrix_pulay, matrix_fd) = fixed_density_variational_fd(
            &system,
            &params,
            &options.electronic,
            &analytic.electronic_result,
            atom,
            axis,
            h,
        );
        let ana_total = component(&analytic.gradient[atom], axis);
        let ana_elec = component(&analytic.electronic_gradient[atom], axis);
        let ana_rep = component(&analytic.repulsion_gradient[atom], axis);
        let ana_disp = component(&analytic.dispersion_gradient[atom], axis);
        let ana_hal = component(&analytic.halogen_gradient[atom], axis);
        let ana_no_scc_elec = component(&no_scc_analytic.electronic_gradient[atom], axis);
        let fd_no_cn_elec = finite_difference(
            &system,
            &params,
            &no_cn_options.electronic,
            atom,
            axis,
            h,
            |e| e.electronic_energy + e.isotropic_scc_energy + e.third_order_energy,
        );
        let ana_no_cn_elec = component(&no_cn_analytic.electronic_gradient[atom], axis);
        let (matrix_no_cn_h0, matrix_no_cn_scc, matrix_no_cn_pulay, matrix_no_cn_fd) =
            fixed_density_variational_fd(
                &system,
                &params,
                &no_cn_options.electronic,
                &no_cn_analytic.electronic_result,
                atom,
                axis,
                h,
            );
        let ana_no_cn_no_scc_elec =
            component(&no_cn_no_scc_analytic.electronic_gradient[atom], axis);
        println!(
            "atom {atom:2} axis {axis}: total ana {ana_total:+.12e} fd {fd_total:+.12e} diff {diff:+.3e}",
            diff = ana_total - fd_total
        );
        println!(
            "  elec {ana_elec:+.12e} fd {fd_elec:+.12e} diff {diff:+.3e}",
            diff = ana_elec - fd_elec
        );
        println!(
            "  rep  {ana_rep:+.12e} fd {fd_rep:+.12e} diff {diff:+.3e}",
            diff = ana_rep - fd_rep
        );
        println!(
            "  disp {ana_disp:+.12e} fd {fd_disp:+.12e} diff {diff:+.3e}",
            diff = ana_disp - fd_disp
        );
        println!(
            "  hal  {ana_hal:+.12e} fd {fd_hal:+.12e} diff {diff:+.3e}",
            diff = ana_hal - fd_hal
        );
        println!(
            "  noSCC grad-elec {ana_no_scc_elec:+.12e} scc-part {scc_part:+.12e}",
            scc_part = ana_elec - ana_no_scc_elec
        );
        println!(
            "  matrix h0 {matrix_h0:+.12e} scc {matrix_scc:+.12e} pulay {matrix_pulay:+.12e} var {matrix_fd:+.12e} diff {diff:+.3e}",
            diff = ana_elec - matrix_fd
        );
        println!(
            "  noCN elec {ana_no_cn_elec:+.12e} fd {fd_no_cn_elec:+.12e} matrix {matrix_no_cn_fd:+.12e} diff {diff:+.3e}",
            diff = ana_no_cn_elec - fd_no_cn_elec
        );
        println!(
            "  noCN noSCC {ana_no_cn_no_scc_elec:+.12e} matrix h0+pulay {matrix_no_cn_h0_pulay:+.12e} scc {matrix_no_cn_scc:+.12e}",
            matrix_no_cn_h0_pulay = matrix_no_cn_h0 + matrix_no_cn_pulay,
        );
    }
}

fn fixed_density_variational_fd(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    base: &gfn1_rs::ElectronicResult,
    atom: usize,
    axis: usize,
    h: f64,
) -> (f64, f64, f64, f64) {
    let mut plus = system.clone();
    let mut minus = system.clone();
    displace(&mut plus, atom, axis, h);
    displace(&mut minus, atom, axis, -h);
    let (h0p, sccp, sp) = fixed_density_energy_and_overlap(&plus, params, options, base);
    let (h0m, sccm, sm) = fixed_density_energy_and_overlap(&minus, params, options, base);
    let h0_fixed = (h0p - h0m) / (2.0 * h);
    let scc_fixed = (sccp - sccm) / (2.0 * h);
    let mut pulay = 0.0;
    for i in 0..sp.rows() {
        for j in 0..sp.cols() {
            let ds = (sp[(i, j)] - sm[(i, j)]) / (2.0 * h);
            pulay -= base.energy_weighted_density[(i, j)] * ds;
        }
    }
    (h0_fixed, scc_fixed, pulay, h0_fixed + scc_fixed + pulay)
}

fn fixed_density_energy_and_overlap(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    base: &gfn1_rs::ElectronicResult,
) -> (f64, f64, gfn1_rs::linalg::Matrix) {
    let basis = gfn1_rs::BasisSet::build(
        system,
        params,
        BasisOptions {
            nprim: options.nprim,
        },
    )
    .unwrap();
    let core = build_h0(system, &basis, params, &options.hamiltonian).unwrap();
    let q = mulliken_shell_charges(&basis, &core.integrals.overlap, &base.density);
    let model = ShellChargeModel::build(system, &basis, params).unwrap();
    let amat = effective_coulomb_matrix(system, &basis, &model);
    let scc = coulomb_energy_potential_from_matrix(&basis, &model, &q, &amat).unwrap();
    let h0_energy = electronic_energy(&core.h0, &base.density);
    let scc_energy = scc.second_order + scc.third_order;
    (h0_energy, scc_energy, core.integrals.overlap)
}

fn finite_difference<F: Fn(&gfn1_rs::ElectronicResult) -> f64>(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    atom: usize,
    axis: usize,
    h: f64,
    f: F,
) -> f64 {
    let mut plus = system.clone();
    let mut minus = system.clone();
    displace(&mut plus, atom, axis, h);
    displace(&mut minus, atom, axis, -h);
    let ep = run_electronic(&plus, params, options.clone()).unwrap();
    let em = run_electronic(&minus, params, options.clone()).unwrap();
    (f(&ep) - f(&em)) / (2.0 * h)
}

fn displace(system: &mut PeriodicSystem, atom: usize, axis: usize, value: f64) {
    match axis {
        0 => system.atoms[atom].position.x += value,
        1 => system.atoms[atom].position.y += value,
        2 => system.atoms[atom].position.z += value,
        _ => unreachable!(),
    }
}

fn component(v: &gfn1_rs::math::Vec3, axis: usize) -> f64 {
    match axis {
        0 => v.x,
        1 => v.y,
        2 => v.z,
        _ => unreachable!(),
    }
}
