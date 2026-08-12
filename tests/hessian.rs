// SPDX-License-Identifier: GPL-3.0-or-later

use gfn1_rs::coordination::{coordination_with_derivatives, CoordinationOptions};
use gfn1_rs::coulomb::coulomb_energy_potential;
use gfn1_rs::cphf::{solve_nonpbc_cpxtb_hessian_response, AoDerivativeOptions, CpxtbOptions};
use gfn1_rs::{
    analytic_gradient, analytic_hessian, fixed_density_cn_h0_hessian, fixed_density_pulay_hessian,
    fixed_shell_charge_scc_hessian, run_electronic, AnalyticGradientOptions,
    AnalyticHessianOptions, ElectronicOptions, ElectronicResult, Gfn1Parameters, PeriodicSystem,
};

#[test]
fn repulsion_only_analytic_hessian_is_symmetric() {
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    let system = PeriodicSystem::from_xyz_str(
        "3\nwater\nO 0.000000 0.000000 0.000000\nH 0.757000 0.586000 0.000000\nH -0.757000 0.586000 0.000000\n",
        0.0,
        false,
    )
    .unwrap();
    let options = AnalyticHessianOptions {
        include_repulsion: true,
        include_fixed_scc: false,
        include_fixed_pulay: false,
        include_fixed_cn_h0: false,
        include_electronic: false,
        include_dispersion: false,
        include_halogen: false,
        electronic_options: ElectronicOptions::default(),
    };
    let result = analytic_hessian(&system, &params, options).unwrap();
    let n = result.hessian.rows();
    let mut max_asym = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            max_asym = max_asym.max((result.hessian[(i, j)] - result.hessian[(j, i)]).abs());
        }
    }
    assert!(max_asym < 1.0e-14);
}

#[test]
fn fixed_scc_analytic_hessian_matches_gradient_finite_difference() {
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    let system = PeriodicSystem::from_xyz_str(
        "3\nwater\nO 0.000000 0.000000 0.000000\nH 0.757000 0.586000 0.000000\nH -0.757000 0.586000 0.000000\n",
        0.0,
        false,
    )
    .unwrap();
    let electronic = run_electronic(&system, &params, ElectronicOptions::default()).unwrap();
    let analytic = fixed_shell_charge_scc_hessian(
        &system,
        &electronic.basis,
        &electronic.shell_charges,
        &params,
    )
    .unwrap();
    let step = 1.0e-4;
    let ndof = 3 * system.atoms.len();
    let mut max_delta = 0.0_f64;
    for col in 0..ndof {
        let mut plus = system.clone();
        let mut minus = system.clone();
        displace(&mut plus, col, step);
        displace(&mut minus, col, -step);
        let gp = fixed_shell_charge_scc_hessian(
            &plus,
            &electronic.basis,
            &electronic.shell_charges,
            &params,
        )
        .unwrap()
        .gradient;
        let gm = fixed_shell_charge_scc_hessian(
            &minus,
            &electronic.basis,
            &electronic.shell_charges,
            &params,
        )
        .unwrap()
        .gradient;
        for row in 0..ndof {
            let fd = (component(&gp, row) - component(&gm, row)) / (2.0 * step);
            max_delta = max_delta.max((analytic.hessian[(row, col)] - fd).abs());
        }
    }
    assert!(
        max_delta < 1.0e-7,
        "fixed SCC Hessian finite-difference max delta {max_delta:.3e}"
    );
}

#[test]
fn fixed_density_pulay_hessian_matches_gradient_finite_difference() {
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    let system = PeriodicSystem::from_xyz_str(
        "2\nHF\nH 0.000000 0.000000 0.000000\nF 0.917000 0.000000 0.000000\n",
        0.0,
        false,
    )
    .unwrap();
    let electronic = run_electronic(&system, &params, ElectronicOptions::default()).unwrap();
    let analytic = fixed_density_pulay_hessian(&system, &params, &electronic).unwrap();
    let step = 1.0e-4;
    let ndof = 3 * system.atoms.len();
    let mut max_delta = 0.0_f64;
    for col in 0..ndof {
        let mut plus = system.clone();
        let mut minus = system.clone();
        displace(&mut plus, col, step);
        displace(&mut minus, col, -step);
        let gp = fixed_density_pulay_hessian(&plus, &params, &electronic)
            .unwrap()
            .gradient;
        let gm = fixed_density_pulay_hessian(&minus, &params, &electronic)
            .unwrap()
            .gradient;
        for row in 0..ndof {
            let fd = (component(&gp, row) - component(&gm, row)) / (2.0 * step);
            max_delta = max_delta.max((analytic.hessian[(row, col)] - fd).abs());
        }
    }
    assert!(
        max_delta < 1.0e-7,
        "fixed-density Pulay Hessian finite-difference max delta {max_delta:.3e}"
    );
}

#[test]
fn fixed_density_cn_h0_hessian_matches_gradient_finite_difference() {
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    let system = PeriodicSystem::from_xyz_str(
        "3\nwater\nO 0.000000 0.000000 0.000000\nH 0.757000 0.586000 0.000000\nH -0.757000 0.586000 0.000000\n",
        0.0,
        false,
    )
    .unwrap();
    let options = ElectronicOptions::default();
    let electronic = run_electronic(&system, &params, options.clone()).unwrap();
    let cutoff = options.hamiltonian.coordination_cutoff;
    let analytic = fixed_density_cn_h0_hessian(&system, &params, &electronic, cutoff).unwrap();
    let step = 1.0e-4;
    let ndof = 3 * system.atoms.len();
    let mut max_delta = 0.0_f64;
    for col in 0..ndof {
        let mut plus = system.clone();
        let mut minus = system.clone();
        displace(&mut plus, col, step);
        displace(&mut minus, col, -step);
        let gp = fixed_density_cn_h0_hessian(&plus, &params, &electronic, cutoff)
            .unwrap()
            .gradient;
        let gm = fixed_density_cn_h0_hessian(&minus, &params, &electronic, cutoff)
            .unwrap()
            .gradient;
        for row in 0..ndof {
            let fd = (component(&gp, row) - component(&gm, row)) / (2.0 * step);
            max_delta = max_delta.max((analytic.hessian[(row, col)] - fd).abs());
        }
    }
    assert!(
        max_delta < 1.0e-7,
        "fixed-density CN-H0 Hessian finite-difference max delta {max_delta:.3e}"
    );
}

#[test]
fn relaxed_electronic_hessian_matches_gradient_finite_difference_h2() {
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    let system = PeriodicSystem::from_xyz_str(
        "2\nH2\nH 0.000000 0.000000 0.000000\nH 0.740000 0.000000 0.000000\n",
        0.0,
        false,
    )
    .unwrap();
    let mut electronic_options = ElectronicOptions::default();
    electronic_options.energy_tolerance = 1.0e-12;
    electronic_options.charge_tolerance = 1.0e-10;
    electronic_options.max_scc = 500;
    let hessian_options = AnalyticHessianOptions {
        include_repulsion: false,
        include_fixed_scc: true,
        include_fixed_pulay: true,
        include_fixed_cn_h0: true,
        include_electronic: true,
        include_dispersion: false,
        include_halogen: false,
        electronic_options: electronic_options.clone(),
    };
    let analytic = analytic_hessian(&system, &params, hessian_options).unwrap();
    assert!(
        analytic
            .cpxtb_response
            .as_ref()
            .is_some_and(|r| r.converged),
        "CPXTB response did not converge"
    );
    let grad_options = AnalyticGradientOptions {
        electronic: electronic_options,
        include_repulsion: false,
        include_dispersion: false,
        include_hamiltonian: true,
        include_scc: true,
        include_halogen: false,
    };
    let step = 1.0e-4;
    let ndof = 3 * system.atoms.len();
    let mut max_delta = 0.0_f64;
    let mut max_entry = (0usize, 0usize, 0.0_f64, 0.0_f64);
    for col in 0..ndof {
        let mut plus = system.clone();
        let mut minus = system.clone();
        displace(&mut plus, col, step);
        displace(&mut minus, col, -step);
        let gp = analytic_gradient(&plus, &params, grad_options.clone())
            .unwrap()
            .electronic_gradient;
        let gm = analytic_gradient(&minus, &params, grad_options.clone())
            .unwrap()
            .electronic_gradient;
        for row in 0..ndof {
            let fd = (component(&gp, row) - component(&gm, row)) / (2.0 * step);
            let delta = (analytic.hessian[(row, col)] - fd).abs();
            if delta > max_delta {
                max_delta = delta;
                max_entry = (row, col, analytic.hessian[(row, col)], fd);
            }
        }
    }
    let (
        response_density_delta,
        response_weighted_delta,
        shell_response_delta,
        potential_response_delta,
        functional_response_delta,
        synthetic_functional_delta,
    ) = {
        let col = max_entry.1;
        let mut plus = system.clone();
        let mut minus = system.clone();
        displace(&mut plus, col, step);
        displace(&mut minus, col, -step);
        let ep = run_electronic(&plus, &params, grad_options.electronic.clone()).unwrap();
        let em = run_electronic(&minus, &params, grad_options.electronic.clone()).unwrap();
        let response = analytic.cpxtb_response.as_ref().unwrap();
        let mut max_density = 0.0_f64;
        let mut max_weighted = 0.0_f64;
        for idx in 0..ep.density.as_slice().len() {
            let fd = (ep.density.as_slice()[idx] - em.density.as_slice()[idx]) / (2.0 * step);
            max_density =
                max_density.max((response.density_responses[col].as_slice()[idx] - fd).abs());
            let wfd = (ep.energy_weighted_density.as_slice()[idx]
                - em.energy_weighted_density.as_slice()[idx])
                / (2.0 * step);
            max_weighted = max_weighted
                .max((response.energy_weighted_density_responses[col].as_slice()[idx] - wfd).abs());
        }
        let mut max_shell = 0.0_f64;
        for idx in 0..ep.shell_charges.len() {
            let fd = (ep.shell_charges[idx] - em.shell_charges[idx]) / (2.0 * step);
            max_shell = max_shell.max((response.shell_charge_responses[col][idx] - fd).abs());
        }
        let kernel = gfn1_rs::cphf::response_shell_scc_kernel(
            &system,
            &params,
            analytic.electronic_result.as_ref().unwrap(),
        )
        .unwrap();
        let response_potential =
            gfn1_rs::linalg::matrix_vector_product(&kernel, &response.shell_charge_responses[col])
                .unwrap();
        let mut max_potential = 0.0_f64;
        for idx in 0..ep.shell_scc_potential.len() {
            let fd = (ep.shell_scc_potential[idx] - em.shell_scc_potential[idx]) / (2.0 * step);
            max_potential = max_potential.max((response_potential[idx] - fd).abs());
        }
        let gp_state =
            gfn1_rs::gradient::analytic_gradient_from_result(&system, &params, ep, &grad_options)
                .unwrap()
                .electronic_gradient;
        let gm_state =
            gfn1_rs::gradient::analytic_gradient_from_result(&system, &params, em, &grad_options)
                .unwrap()
                .electronic_gradient;
        let mut max_functional = 0.0_f64;
        for row in 0..ndof {
            let fd = (component(&gp_state, row) - component(&gm_state, row)) / (2.0 * step);
            max_functional = max_functional.max((response.hessian_response[(row, col)] - fd).abs());
        }
        let base = analytic.electronic_result.as_ref().unwrap();
        let mut esp = base.clone();
        let mut esm = base.clone();
        let eps = 1.0e-5;
        for idx in 0..base.density.as_slice().len() {
            esp.density.as_mut_slice()[idx] +=
                eps * response.density_responses[col].as_slice()[idx];
            esm.density.as_mut_slice()[idx] -=
                eps * response.density_responses[col].as_slice()[idx];
            esp.energy_weighted_density.as_mut_slice()[idx] +=
                eps * response.energy_weighted_density_responses[col].as_slice()[idx];
            esm.energy_weighted_density.as_mut_slice()[idx] -=
                eps * response.energy_weighted_density_responses[col].as_slice()[idx];
        }
        for idx in 0..base.shell_charges.len() {
            esp.shell_charges[idx] += eps * response.shell_charge_responses[col][idx];
            esm.shell_charges[idx] -= eps * response.shell_charge_responses[col][idx];
            esp.shell_scc_potential[idx] += eps * response_potential[idx];
            esm.shell_scc_potential[idx] -= eps * response_potential[idx];
        }
        let gp_synth =
            gfn1_rs::gradient::analytic_gradient_from_result(&system, &params, esp, &grad_options)
                .unwrap()
                .electronic_gradient;
        let gm_synth =
            gfn1_rs::gradient::analytic_gradient_from_result(&system, &params, esm, &grad_options)
                .unwrap()
                .electronic_gradient;
        let mut max_synthetic = 0.0_f64;
        for row in 0..ndof {
            let fd = (component(&gp_synth, row) - component(&gm_synth, row)) / (2.0 * eps);
            max_synthetic = max_synthetic.max((response.hessian_response[(row, col)] - fd).abs());
        }
        (
            max_density,
            max_weighted,
            max_shell,
            max_potential,
            max_functional,
            max_synthetic,
        )
    };
    let (direct_fixed_delta, direct_functional_delta, direct_entry_parts) = {
        let electronic = analytic.electronic_result.as_ref().unwrap();
        let cutoff = grad_options.electronic.hamiltonian.coordination_cutoff;
        let mut direct = gfn1_rs::linalg::Matrix::zeros(ndof, ndof);
        let scc = fixed_shell_charge_scc_hessian(
            &system,
            &electronic.basis,
            &electronic.shell_charges,
            &params,
        )
        .unwrap();
        let pulay = fixed_density_pulay_hessian(&system, &params, electronic).unwrap();
        let cn = fixed_density_cn_h0_hessian(&system, &params, electronic, cutoff).unwrap();
        for idx in 0..direct.as_mut_slice().len() {
            direct.as_mut_slice()[idx] = scc.hessian.as_slice()[idx]
                + pulay.hessian.as_slice()[idx]
                + cn.hessian.as_slice()[idx];
        }
        let mut max = 0.0_f64;
        for col in 0..ndof {
            let mut plus = system.clone();
            let mut minus = system.clone();
            displace(&mut plus, col, step);
            displace(&mut minus, col, -step);
            let sccp = fixed_shell_charge_scc_hessian(
                &plus,
                &electronic.basis,
                &electronic.shell_charges,
                &params,
            )
            .unwrap()
            .gradient;
            let sccm = fixed_shell_charge_scc_hessian(
                &minus,
                &electronic.basis,
                &electronic.shell_charges,
                &params,
            )
            .unwrap()
            .gradient;
            let pp = fixed_density_pulay_hessian(&plus, &params, electronic)
                .unwrap()
                .gradient;
            let pm = fixed_density_pulay_hessian(&minus, &params, electronic)
                .unwrap()
                .gradient;
            let cnp = fixed_density_cn_h0_hessian(&plus, &params, electronic, cutoff)
                .unwrap()
                .gradient;
            let cnm = fixed_density_cn_h0_hessian(&minus, &params, electronic, cutoff)
                .unwrap()
                .gradient;
            for row in 0..ndof {
                let gp = component(&sccp, row) + component(&pp, row) + component(&cnp, row);
                let gm = component(&sccm, row) + component(&pm, row) + component(&cnm, row);
                let fd = (gp - gm) / (2.0 * step);
                max = max.max((direct[(row, col)] - fd).abs());
            }
        }
        let mut max_direct_functional = 0.0_f64;
        for col in 0..ndof {
            let mut plus = system.clone();
            let mut minus = system.clone();
            displace(&mut plus, col, step);
            displace(&mut minus, col, -step);
            let gp = gfn1_rs::gradient::analytic_gradient_from_result(
                &plus,
                &params,
                electronic.clone(),
                &grad_options,
            )
            .unwrap()
            .electronic_gradient;
            let gm = gfn1_rs::gradient::analytic_gradient_from_result(
                &minus,
                &params,
                electronic.clone(),
                &grad_options,
            )
            .unwrap()
            .electronic_gradient;
            for row in 0..ndof {
                let fd = (component(&gp, row) - component(&gm, row)) / (2.0 * step);
                max_direct_functional = max_direct_functional.max((direct[(row, col)] - fd).abs());
            }
        }
        (
            max,
            max_direct_functional,
            (
                scc.hessian[(max_entry.0, max_entry.1)],
                pulay.hessian[(max_entry.0, max_entry.1)],
                cn.hessian[(max_entry.0, max_entry.1)],
            ),
        )
    };
    assert!(
        max_delta < 1.0e-6,
        "relaxed electronic Hessian finite-difference max delta {max_delta:.3e}, entry {max_entry:?}, cphf {:?}, density_response_delta {response_density_delta:.3e}, weighted_response_delta {response_weighted_delta:.3e}, shell_response_delta {shell_response_delta:.3e}, potential_response_delta {potential_response_delta:.3e}, functional_response_delta {functional_response_delta:.3e}, synthetic_functional_delta {synthetic_functional_delta:.3e}, direct_fixed_delta {direct_fixed_delta:.3e}, direct_functional_delta {direct_functional_delta:.3e}, direct_entry_parts {direct_entry_parts:?}",
        analytic
            .cpxtb_response
            .as_ref()
            .map(|r| (
                r.hessian_response[(max_entry.0, max_entry.1)],
                r.shell_charge_responses[max_entry.1].clone()
            ))
    );
}

#[test]
fn relaxed_electronic_hessian_matches_gradient_finite_difference_water() {
    assert_relaxed_electronic_hessian_matches_gradient_finite_difference_for_xyz(
        "water",
        "3\nwater\nO 0.000000 0.000000 0.000000\nH 0.757000 0.586000 0.000000\nH -0.757000 0.586000 0.000000\n",
        1.0e-6,
    );
}

#[test]
fn relaxed_electronic_hessian_matches_gradient_finite_difference_ammonia() {
    assert_relaxed_electronic_hessian_matches_gradient_finite_difference_for_xyz(
        "ammonia",
        "4\nammonia\nN 0.000000 0.000000 0.100000\nH 0.000000 0.942000 -0.267000\nH 0.816000 -0.471000 -0.267000\nH -0.816000 -0.471000 -0.267000\n",
        1.0e-6,
    );
}

#[test]
fn relaxed_electronic_hessian_matches_gradient_finite_difference_chloromethanol() {
    assert_relaxed_electronic_hessian_matches_gradient_finite_difference_for_xyz(
        "chloromethanol",
        "6\nchloromethanol\nC 0.000000 0.000000 0.000000\nO 1.410000 0.000000 0.000000\nCl -1.760000 0.000000 0.000000\nH 0.030000 1.020000 0.000000\nH 0.030000 -0.510000 0.884000\nH 1.780000 0.000000 0.890000\n",
        1.0e-6,
    );
}

#[test]
fn relaxed_electronic_hessian_matches_gradient_finite_difference_bromoethanol() {
    assert_relaxed_electronic_hessian_matches_gradient_finite_difference_for_xyz(
        "bromoethanol",
        "9\nbromoethanol\nC 0.000000 0.000000 0.000000\nC 1.520000 0.000000 0.000000\nO 2.160000 1.220000 0.000000\nBr -1.940000 0.000000 0.000000\nH 0.220000 1.020000 0.000000\nH 0.220000 -0.510000 0.884000\nH 1.880000 -0.510000 -0.884000\nH 1.880000 -0.510000 0.884000\nH 2.960000 1.100000 0.500000\n",
        1.0e-6,
    );
}

#[test]
fn relaxed_electronic_hessian_matches_gradient_finite_difference_ni_co4() {
    assert_relaxed_electronic_hessian_matches_gradient_finite_difference_for_xyz(
        "Ni(CO)4",
        "9\nNi(CO)4\nNi 0.000000 0.000000 0.000000\nC 1.820000 1.820000 1.820000\nO 2.480000 2.480000 2.480000\nC -1.820000 -1.820000 1.820000\nO -2.480000 -2.480000 2.480000\nC -1.820000 1.820000 -1.820000\nO -2.480000 2.480000 -2.480000\nC 1.820000 -1.820000 -1.820000\nO 2.480000 -2.480000 -2.480000\n",
        1.0e-6,
    );
}

#[test]
fn optimized_cpxtb_response_matches_scc_finite_difference_water_selected_columns() {
    assert_cpxtb_response_matches_scc_finite_difference_for_xyz(
        "water optimized CPXTB response",
        "3\nwater\nO 0.000000 0.000000 0.000000\nH 0.757000 0.586000 0.000000\nH -0.757000 0.586000 0.000000\n",
        &[0, 4, 8],
        2.0e-6,
        2.0e-6,
        2.0e-6,
        2.0e-6,
    );
}

#[test]
fn optimized_finite_temperature_cpxtb_response_matches_scc_finite_difference_ni_co4_selected_columns(
) {
    assert_cpxtb_response_matches_scc_finite_difference_for_xyz(
        "Ni(CO)4 finite-temperature optimized CPXTB response",
        "9\nNi(CO)4\nNi 0.000000 0.000000 0.000000\nC 1.820000 1.820000 1.820000\nO 2.480000 2.480000 2.480000\nC -1.820000 -1.820000 1.820000\nO -2.480000 -2.480000 2.480000\nC -1.820000 1.820000 -1.820000\nO -2.480000 2.480000 -2.480000\nC 1.820000 -1.820000 -1.820000\nO 2.480000 -2.480000 -2.480000\n",
        &[0, 13, 26],
        5.0e-6,
        5.0e-6,
        5.0e-6,
        5.0e-6,
    );
}

/// v0.5.0 regression: with `charge_order = 4` (Linear Breathing-Radius on-site
/// orders) the CPXTB response kernel must carry the anharmonic
/// `Σ_{n≥4}(n−1)X_n q^{n−2}` on-site block — before the fix the kernel was
/// silently truncated at the DFTB3 `2Γq` term, so the relaxed Hessian belonged
/// to a different energy expression than the gradient it should differentiate.
#[test]
fn relaxed_electronic_hessian_matches_gradient_fd_charge_order_4() {
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    let system = PeriodicSystem::from_xyz_str(
        "3\nwater\nO 0.000000 0.000000 0.000000\nH 0.757000 0.586000 0.000000\nH -0.757000 0.586000 0.000000\n",
        0.0,
        false,
    )
    .unwrap();
    let mut electronic_options = ElectronicOptions::default();
    electronic_options.energy_tolerance = 1.0e-12;
    electronic_options.charge_tolerance = 1.0e-10;
    electronic_options.max_scc = 500;
    electronic_options.charge_order = 4;
    let hessian_options = AnalyticHessianOptions {
        include_repulsion: false,
        include_dispersion: false,
        include_halogen: false,
        electronic_options: electronic_options.clone(),
        ..AnalyticHessianOptions::default()
    };
    let analytic = analytic_hessian(&system, &params, hessian_options).unwrap();
    let grad_options = AnalyticGradientOptions {
        electronic: electronic_options,
        include_repulsion: false,
        include_dispersion: false,
        include_hamiltonian: true,
        include_scc: true,
        include_halogen: false,
    };
    let step = 1.0e-5;
    let ndof = 3 * system.atoms.len();
    let mut max_delta = 0.0_f64;
    for col in 0..ndof {
        let mut plus = system.clone();
        let mut minus = system.clone();
        displace(&mut plus, col, step);
        displace(&mut minus, col, -step);
        let gp = analytic_gradient(&plus, &params, grad_options.clone())
            .unwrap()
            .electronic_gradient;
        let gm = analytic_gradient(&minus, &params, grad_options.clone())
            .unwrap()
            .electronic_gradient;
        for row in 0..ndof {
            let fd = (component(&gp, row) - component(&gm, row)) / (2.0 * step);
            max_delta = max_delta.max((analytic.hessian[(row, col)] - fd).abs());
        }
    }
    assert!(
        max_delta < 1.0e-6,
        "charge_order=4 relaxed Hessian vs gradient FD: max delta {max_delta:.3e}"
    );
}

/// v0.5.0 regression: the analytic Hessian must REJECT option sets whose
/// second-derivative terms it does not implement (multipole, exchange, +U,
/// spin polarization, D4, external field) instead of silently dropping them.
#[test]
fn analytic_hessian_rejects_unsupported_terms() {
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    let system = PeriodicSystem::from_xyz_str(
        "3\nwater\nO 0.000000 0.000000 0.000000\nH 0.757000 0.586000 0.000000\nH -0.757000 0.586000 0.000000\n",
        0.0,
        false,
    )
    .unwrap();
    let cases: Vec<(&str, ElectronicOptions)> = vec![
        (
            "multipole",
            ElectronicOptions {
                multipole: true,
                ..ElectronicOptions::default()
            },
        ),
        (
            "lr_exchange",
            ElectronicOptions {
                lr_exchange: true,
                ..ElectronicOptions::default()
            },
        ),
        (
            "plus_u",
            ElectronicOptions {
                plus_u: true,
                ..ElectronicOptions::default()
            },
        ),
        (
            "spin_polarization",
            ElectronicOptions {
                spin_polarization: true,
                ..ElectronicOptions::default()
            },
        ),
        (
            "experimental_d4",
            ElectronicOptions {
                experimental_d4: true,
                ..ElectronicOptions::default()
            },
        ),
        (
            "external_field",
            ElectronicOptions {
                external_field: gfn1_rs::field::ExternalFieldOptions::electric(
                    gfn1_rs::math::Vec3::new(0.0, 0.0, 1.0e-3),
                ),
                ..ElectronicOptions::default()
            },
        ),
    ];
    for (label, electronic_options) in cases {
        let options = AnalyticHessianOptions {
            electronic_options,
            ..AnalyticHessianOptions::default()
        };
        let err = analytic_hessian(&system, &params, options);
        assert!(
            err.is_err(),
            "{label}: analytic_hessian should reject this unsupported option"
        );
        let msg = format!("{}", err.err().unwrap());
        assert!(
            msg.contains("requires analytic order-2"),
            "{label}: unexpected error message: {msg}"
        );
    }
}

/// v0.5.0 regression: a zero-gap (open-shell-degenerate) configuration with
/// integer occupations makes the CPXTB operator singular; the solver used to
/// return ~1e42 garbage without erroring. It must now reject with a clear
/// message. The system is CH3Br plus a BARE oxygen atom (a symmetry-broken
/// aufbau filling of O's degenerate 2p shell), which converges in the SCC but
/// has a vanishing occupied-virtual gap.
#[test]
fn analytic_hessian_rejects_zero_gap_integer_occupations() {
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    let system = PeriodicSystem::from_xyz_str(
        "6\nCH3Br + bare O\nC 0.000000 0.000000 0.000000\nBr 0.000000 0.000000 1.950000\nH 1.030000 0.000000 -0.330000\nH -0.515000 0.892000 -0.330000\nH -0.515000 -0.892000 -0.330000\nO 0.000000 0.100000 4.900000\n",
        0.0,
        false,
    )
    .unwrap();
    let options = AnalyticHessianOptions::default();
    let result = analytic_hessian(&system, &params, options);
    match result {
        Err(err) => {
            let msg = format!("{err}");
            assert!(
                msg.contains("singular") || msg.contains("degenerate"),
                "unexpected error message: {msg}"
            );
        }
        Ok(res) => {
            // If a future SCC finds a gapped solution instead, the Hessian must
            // at least be sane — the historical failure mode was ~1e42 entries.
            let max = res
                .hessian
                .as_slice()
                .iter()
                .fold(0.0_f64, |m, v| m.max(v.abs()));
            assert!(
                max < 1.0e6,
                "zero-gap system returned a garbage Hessian (max entry {max:.3e})"
            );
        }
    }
}

fn assert_relaxed_electronic_hessian_matches_gradient_finite_difference_for_xyz(
    name: &str,
    xyz: &str,
    threshold: f64,
) {
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
    let mut electronic_options = ElectronicOptions::default();
    electronic_options.energy_tolerance = 1.0e-12;
    electronic_options.charge_tolerance = 1.0e-10;
    electronic_options.max_scc = 500;
    let hessian_options = AnalyticHessianOptions {
        include_repulsion: false,
        include_fixed_scc: true,
        include_fixed_pulay: true,
        include_fixed_cn_h0: true,
        include_electronic: true,
        include_dispersion: false,
        include_halogen: false,
        electronic_options: electronic_options.clone(),
    };
    let analytic = analytic_hessian(&system, &params, hessian_options).unwrap();
    assert!(
        analytic
            .cpxtb_response
            .as_ref()
            .is_some_and(|r| r.converged),
        "{name}: CPXTB response did not converge"
    );
    let grad_options = AnalyticGradientOptions {
        electronic: electronic_options,
        include_repulsion: false,
        include_dispersion: false,
        include_hamiltonian: true,
        include_scc: true,
        include_halogen: false,
    };
    let step = 1.0e-5;
    let ndof = 3 * system.atoms.len();
    let mut max_delta = 0.0_f64;
    let mut max_entry = (0usize, 0usize, 0.0_f64, 0.0_f64);
    for col in 0..ndof {
        let mut plus = system.clone();
        let mut minus = system.clone();
        displace(&mut plus, col, step);
        displace(&mut minus, col, -step);
        let gp = analytic_gradient(&plus, &params, grad_options.clone())
            .unwrap()
            .electronic_gradient;
        let gm = analytic_gradient(&minus, &params, grad_options.clone())
            .unwrap()
            .electronic_gradient;
        for row in 0..ndof {
            let fd = (component(&gp, row) - component(&gm, row)) / (2.0 * step);
            let delta = (analytic.hessian[(row, col)] - fd).abs();
            if delta > max_delta {
                max_delta = delta;
                max_entry = (row, col, analytic.hessian[(row, col)], fd);
            }
        }
    }
    let (
        response_density_delta,
        response_weighted_delta,
        shell_response_delta,
        potential_response_delta,
        occupation_delta,
        occupation_response_delta,
        synthetic_functional_delta,
    ) = {
        let col = max_entry.1;
        let mut plus = system.clone();
        let mut minus = system.clone();
        displace(&mut plus, col, step);
        displace(&mut minus, col, -step);
        let ep = run_electronic(&plus, &params, grad_options.electronic.clone()).unwrap();
        let em = run_electronic(&minus, &params, grad_options.electronic.clone()).unwrap();
        let response = analytic.cpxtb_response.as_ref().unwrap();
        let mut max_occ = 0.0_f64;
        let mut max_occ_response = 0.0_f64;
        for idx in 0..ep.occupations.len() {
            let fd = (ep.occupations[idx] - em.occupations[idx]) / (2.0 * step);
            max_occ = max_occ.max(fd.abs());
            max_occ_response =
                max_occ_response.max((response.occupation_responses[col][idx] - fd).abs());
        }
        let mut max_density = 0.0_f64;
        let mut max_weighted = 0.0_f64;
        for idx in 0..ep.density.as_slice().len() {
            let fd = (ep.density.as_slice()[idx] - em.density.as_slice()[idx]) / (2.0 * step);
            max_density =
                max_density.max((response.density_responses[col].as_slice()[idx] - fd).abs());
            let wfd = (ep.energy_weighted_density.as_slice()[idx]
                - em.energy_weighted_density.as_slice()[idx])
                / (2.0 * step);
            max_weighted = max_weighted
                .max((response.energy_weighted_density_responses[col].as_slice()[idx] - wfd).abs());
        }
        let mut max_shell = 0.0_f64;
        for idx in 0..ep.shell_charges.len() {
            let fd = (ep.shell_charges[idx] - em.shell_charges[idx]) / (2.0 * step);
            max_shell = max_shell.max((response.shell_charge_responses[col][idx] - fd).abs());
        }
        let kernel = gfn1_rs::cphf::response_shell_scc_kernel(
            &system,
            &params,
            analytic.electronic_result.as_ref().unwrap(),
        )
        .unwrap();
        let response_potential =
            gfn1_rs::linalg::matrix_vector_product(&kernel, &response.shell_charge_responses[col])
                .unwrap();
        let mut max_potential = 0.0_f64;
        for idx in 0..ep.shell_scc_potential.len() {
            let fd = (ep.shell_scc_potential[idx] - em.shell_scc_potential[idx]) / (2.0 * step);
            max_potential = max_potential.max((response_potential[idx] - fd).abs());
        }
        let base = analytic.electronic_result.as_ref().unwrap();
        let mut esp = base.clone();
        let mut esm = base.clone();
        let eps = 1.0e-5;
        for idx in 0..base.density.as_slice().len() {
            esp.density.as_mut_slice()[idx] +=
                eps * response.density_responses[col].as_slice()[idx];
            esm.density.as_mut_slice()[idx] -=
                eps * response.density_responses[col].as_slice()[idx];
            esp.energy_weighted_density.as_mut_slice()[idx] +=
                eps * response.energy_weighted_density_responses[col].as_slice()[idx];
            esm.energy_weighted_density.as_mut_slice()[idx] -=
                eps * response.energy_weighted_density_responses[col].as_slice()[idx];
        }
        for idx in 0..base.shell_charges.len() {
            esp.shell_charges[idx] += eps * response.shell_charge_responses[col][idx];
            esm.shell_charges[idx] -= eps * response.shell_charge_responses[col][idx];
            esp.shell_scc_potential[idx] += eps * response_potential[idx];
            esm.shell_scc_potential[idx] -= eps * response_potential[idx];
        }
        let gp_synth =
            gfn1_rs::gradient::analytic_gradient_from_result(&system, &params, esp, &grad_options)
                .unwrap()
                .electronic_gradient;
        let gm_synth =
            gfn1_rs::gradient::analytic_gradient_from_result(&system, &params, esm, &grad_options)
                .unwrap()
                .electronic_gradient;
        let mut max_synthetic = 0.0_f64;
        for row in 0..ndof {
            let fd = (component(&gp_synth, row) - component(&gm_synth, row)) / (2.0 * eps);
            max_synthetic = max_synthetic.max((response.hessian_response[(row, col)] - fd).abs());
        }
        (
            max_density,
            max_weighted,
            max_shell,
            max_potential,
            max_occ,
            max_occ_response,
            max_synthetic,
        )
    };
    let entry_parts = {
        let row = max_entry.0;
        let col = max_entry.1;
        let scc = analytic
            .fixed_scc
            .as_ref()
            .map(|v| v.hessian[(row, col)])
            .unwrap_or(0.0);
        let pulay = analytic
            .fixed_pulay
            .as_ref()
            .map(|v| v.hessian[(row, col)])
            .unwrap_or(0.0);
        let cn = analytic
            .fixed_cn_h0
            .as_ref()
            .map(|v| v.hessian[(row, col)])
            .unwrap_or(0.0);
        let cphf = analytic
            .cpxtb_response
            .as_ref()
            .map(|v| v.hessian_response[(row, col)])
            .unwrap_or(0.0);
        let accounted = scc + pulay + cn + cphf;
        (
            scc,
            pulay,
            cn,
            cphf,
            analytic.hessian[(row, col)] - accounted,
        )
    };
    let (
        direct_fixed_delta,
        direct_functional_delta,
        fixed_geometry_delta,
        fixed_geometry_entry_parts,
    ) = {
        let electronic = analytic.electronic_result.as_ref().unwrap();
        let cutoff = grad_options.electronic.hamiltonian.coordination_cutoff;
        let mut direct = gfn1_rs::linalg::Matrix::zeros(ndof, ndof);
        let scc = fixed_shell_charge_scc_hessian(
            &system,
            &electronic.basis,
            &electronic.shell_charges,
            &params,
        )
        .unwrap();
        let pulay = fixed_density_pulay_hessian(&system, &params, electronic).unwrap();
        let cn = fixed_density_cn_h0_hessian(&system, &params, electronic, cutoff).unwrap();
        for idx in 0..direct.as_mut_slice().len() {
            direct.as_mut_slice()[idx] = scc.hessian.as_slice()[idx]
                + pulay.hessian.as_slice()[idx]
                + cn.hessian.as_slice()[idx];
        }
        let mut max_direct = 0.0_f64;
        for col in 0..ndof {
            let mut plus = system.clone();
            let mut minus = system.clone();
            displace(&mut plus, col, step);
            displace(&mut minus, col, -step);
            let sccp = fixed_shell_charge_scc_hessian(
                &plus,
                &electronic.basis,
                &electronic.shell_charges,
                &params,
            )
            .unwrap()
            .gradient;
            let sccm = fixed_shell_charge_scc_hessian(
                &minus,
                &electronic.basis,
                &electronic.shell_charges,
                &params,
            )
            .unwrap()
            .gradient;
            let pp = fixed_density_pulay_hessian(&plus, &params, electronic)
                .unwrap()
                .gradient;
            let pm = fixed_density_pulay_hessian(&minus, &params, electronic)
                .unwrap()
                .gradient;
            let cnp = fixed_density_cn_h0_hessian(&plus, &params, electronic, cutoff)
                .unwrap()
                .gradient;
            let cnm = fixed_density_cn_h0_hessian(&minus, &params, electronic, cutoff)
                .unwrap()
                .gradient;
            for row in 0..ndof {
                let gp = component(&sccp, row) + component(&pp, row) + component(&cnp, row);
                let gm = component(&sccm, row) + component(&pm, row) + component(&cnm, row);
                let fd = (gp - gm) / (2.0 * step);
                max_direct = max_direct.max((direct[(row, col)] - fd).abs());
            }
        }
        let mut max_functional = 0.0_f64;
        for col in 0..ndof {
            let mut plus = system.clone();
            let mut minus = system.clone();
            displace(&mut plus, col, step);
            displace(&mut minus, col, -step);
            let gp = gfn1_rs::gradient::analytic_gradient_from_result(
                &plus,
                &params,
                electronic.clone(),
                &grad_options,
            )
            .unwrap()
            .electronic_gradient;
            let gm = gfn1_rs::gradient::analytic_gradient_from_result(
                &minus,
                &params,
                electronic.clone(),
                &grad_options,
            )
            .unwrap()
            .electronic_gradient;
            for row in 0..ndof {
                let fd = (component(&gp, row) - component(&gm, row)) / (2.0 * step);
                max_functional = max_functional.max((direct[(row, col)] - fd).abs());
            }
        }
        let response = analytic.cpxtb_response.as_ref().unwrap();
        let mut max_fixed_geometry = 0.0_f64;
        for col in 0..ndof {
            let mut plus = system.clone();
            let mut minus = system.clone();
            displace(&mut plus, col, step);
            displace(&mut minus, col, -step);
            let ep =
                fixed_electronic_state_for_geometry(&plus, &params, electronic, cutoff).unwrap();
            let em =
                fixed_electronic_state_for_geometry(&minus, &params, electronic, cutoff).unwrap();
            let gp =
                gfn1_rs::gradient::analytic_gradient_from_result(&plus, &params, ep, &grad_options)
                    .unwrap()
                    .electronic_gradient;
            let gm = gfn1_rs::gradient::analytic_gradient_from_result(
                &minus,
                &params,
                em,
                &grad_options,
            )
            .unwrap()
            .electronic_gradient;
            for row in 0..ndof {
                let fd = (component(&gp, row) - component(&gm, row)) / (2.0 * step);
                let fixed = analytic.hessian[(row, col)] - response.hessian_response[(row, col)];
                max_fixed_geometry = max_fixed_geometry.max((fixed - fd).abs());
            }
        }
        let entry_fd = |update_cn: bool, update_potential: bool| -> f64 {
            let row = max_entry.0;
            let col = max_entry.1;
            let mut plus = system.clone();
            let mut minus = system.clone();
            displace(&mut plus, col, step);
            displace(&mut minus, col, -step);
            let ep = fixed_electronic_state_for_geometry_with(
                &plus,
                &params,
                electronic,
                cutoff,
                update_cn,
                update_potential,
            )
            .unwrap();
            let em = fixed_electronic_state_for_geometry_with(
                &minus,
                &params,
                electronic,
                cutoff,
                update_cn,
                update_potential,
            )
            .unwrap();
            let gp =
                gfn1_rs::gradient::analytic_gradient_from_result(&plus, &params, ep, &grad_options)
                    .unwrap()
                    .electronic_gradient;
            let gm = gfn1_rs::gradient::analytic_gradient_from_result(
                &minus,
                &params,
                em,
                &grad_options,
            )
            .unwrap()
            .electronic_gradient;
            (component(&gp, row) - component(&gm, row)) / (2.0 * step)
        };
        let analytic_fixed_entry = analytic.hessian[(max_entry.0, max_entry.1)]
            - response.hessian_response[(max_entry.0, max_entry.1)];
        let base_entry = entry_fd(false, false);
        let cn_entry = entry_fd(true, false);
        let potential_entry = entry_fd(false, true);
        let full_entry = entry_fd(true, true);
        (
            max_direct,
            max_functional,
            max_fixed_geometry,
            (
                analytic_fixed_entry,
                base_entry,
                cn_entry - base_entry,
                potential_entry - base_entry,
                full_entry,
            ),
        )
    };
    assert!(
        max_delta < threshold,
        "{name}: relaxed electronic Hessian finite-difference max delta {max_delta:.3e}, entry {max_entry:?}, threshold {threshold:.3e}, entry_parts {entry_parts:?}, density_response_delta {response_density_delta:.3e}, weighted_response_delta {response_weighted_delta:.3e}, shell_response_delta {shell_response_delta:.3e}, potential_response_delta {potential_response_delta:.3e}, occupation_delta {occupation_delta:.3e}, occupation_response_delta {occupation_response_delta:.3e}, synthetic_functional_delta {synthetic_functional_delta:.3e}, direct_fixed_delta {direct_fixed_delta:.3e}, direct_functional_delta {direct_functional_delta:.3e}, fixed_geometry_delta {fixed_geometry_delta:.3e}, fixed_geometry_entry_parts {fixed_geometry_entry_parts:?}"
    );
}

fn assert_cpxtb_response_matches_scc_finite_difference_for_xyz(
    name: &str,
    xyz: &str,
    columns: &[usize],
    density_threshold: f64,
    weighted_threshold: f64,
    shell_threshold: f64,
    occupation_threshold: f64,
) {
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
    let mut electronic_options = ElectronicOptions::default();
    electronic_options.energy_tolerance = 1.0e-12;
    electronic_options.charge_tolerance = 1.0e-10;
    electronic_options.max_scc = 500;
    let electronic = run_electronic(&system, &params, electronic_options.clone()).unwrap();
    let response = solve_nonpbc_cpxtb_hessian_response(
        &system,
        &params,
        &electronic,
        AoDerivativeOptions {
            coordination_cutoff: electronic_options.hamiltonian.coordination_cutoff,
            include_cn_h0: electronic_options.hamiltonian.enable_cn_hamiltonian,
        },
        CpxtbOptions::default(),
    )
    .unwrap();
    assert!(
        response.converged,
        "{name}: CPXTB response did not converge"
    );

    let ndof = 3 * system.atoms.len();
    let step = 1.0e-5;
    for &col in columns {
        assert!(
            col < ndof,
            "{name}: selected FD column {col} is out of range"
        );
        let mut plus = system.clone();
        let mut minus = system.clone();
        displace(&mut plus, col, step);
        displace(&mut minus, col, -step);
        let ep = run_electronic(&plus, &params, electronic_options.clone()).unwrap();
        let em = run_electronic(&minus, &params, electronic_options.clone()).unwrap();

        let mut density_delta = 0.0_f64;
        let mut weighted_delta = 0.0_f64;
        for idx in 0..ep.density.as_slice().len() {
            let fd = (ep.density.as_slice()[idx] - em.density.as_slice()[idx]) / (2.0 * step);
            density_delta =
                density_delta.max((response.density_responses[col].as_slice()[idx] - fd).abs());
            let wfd = (ep.energy_weighted_density.as_slice()[idx]
                - em.energy_weighted_density.as_slice()[idx])
                / (2.0 * step);
            weighted_delta = weighted_delta
                .max((response.energy_weighted_density_responses[col].as_slice()[idx] - wfd).abs());
        }

        let mut shell_delta = 0.0_f64;
        for idx in 0..ep.shell_charges.len() {
            let fd = (ep.shell_charges[idx] - em.shell_charges[idx]) / (2.0 * step);
            shell_delta = shell_delta.max((response.shell_charge_responses[col][idx] - fd).abs());
        }

        let occupation_delta = occupation_cluster_response_delta(
            &electronic.orbital_energies,
            &response.occupation_responses[col],
            &ep.occupations,
            &em.occupations,
            step,
        );

        assert!(
            density_delta < density_threshold,
            "{name}: density response FD column {col} delta {density_delta:.3e} exceeds {density_threshold:.3e}"
        );
        assert!(
            weighted_delta < weighted_threshold,
            "{name}: energy-weighted response FD column {col} delta {weighted_delta:.3e} exceeds {weighted_threshold:.3e}"
        );
        assert!(
            shell_delta < shell_threshold,
            "{name}: shell response FD column {col} delta {shell_delta:.3e} exceeds {shell_threshold:.3e}"
        );
        assert!(
            occupation_delta < occupation_threshold,
            "{name}: cluster-summed occupation response FD column {col} delta {occupation_delta:.3e} exceeds {occupation_threshold:.3e}"
        );
    }
}

fn occupation_cluster_response_delta(
    orbital_energies: &[f64],
    analytic_response: &[f64],
    plus_occupations: &[f64],
    minus_occupations: &[f64],
    step: f64,
) -> f64 {
    assert_eq!(orbital_energies.len(), analytic_response.len());
    assert_eq!(orbital_energies.len(), plus_occupations.len());
    assert_eq!(orbital_energies.len(), minus_occupations.len());
    let cluster_tol = 2.0e-5_f64;
    let mut max_delta = 0.0_f64;
    let mut start = 0usize;
    while start < orbital_energies.len() {
        let mut end = start + 1;
        while end < orbital_energies.len()
            && (orbital_energies[end] - orbital_energies[end - 1]).abs() <= cluster_tol
        {
            end += 1;
        }
        let analytic = analytic_response[start..end].iter().sum::<f64>();
        let fd = plus_occupations[start..end]
            .iter()
            .zip(minus_occupations[start..end].iter())
            .map(|(&plus, &minus)| (plus - minus) / (2.0 * step))
            .sum::<f64>();
        max_delta = max_delta.max((analytic - fd).abs());
        start = end;
    }
    max_delta
}

fn displace(system: &mut PeriodicSystem, dof: usize, step: f64) {
    let atom = dof / 3;
    match dof % 3 {
        0 => system.atoms[atom].position.x += step,
        1 => system.atoms[atom].position.y += step,
        _ => system.atoms[atom].position.z += step,
    }
}

fn fixed_electronic_state_for_geometry(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    base: &ElectronicResult,
    coordination_cutoff: f64,
) -> gfn1_rs::Result<ElectronicResult> {
    fixed_electronic_state_for_geometry_with(system, params, base, coordination_cutoff, true, true)
}

fn fixed_electronic_state_for_geometry_with(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    base: &ElectronicResult,
    coordination_cutoff: f64,
    update_cn: bool,
    update_potential: bool,
) -> gfn1_rs::Result<ElectronicResult> {
    let mut state = base.clone();
    if update_cn {
        state.coordination_numbers = coordination_with_derivatives(
            system,
            CoordinationOptions {
                cutoff: coordination_cutoff,
                ..CoordinationOptions::default()
            },
        )?
        .cn;
    }
    if update_potential {
        state.shell_scc_potential =
            coulomb_energy_potential(system, &state.basis, &state.shell_charges, params)?
                .shell_potential;
    }
    Ok(state)
}

fn component(values: &[gfn1_rs::math::Vec3], dof: usize) -> f64 {
    let atom = dof / 3;
    match dof % 3 {
        0 => values[atom].x,
        1 => values[atom].y,
        _ => values[atom].z,
    }
}
