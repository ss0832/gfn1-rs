// SPDX-License-Identifier: GPL-3.0-or-later
//! Regression for the reported PBC k-point Hessian blow-up on bromoethanol.
//! The analytic k-point Hessian must be finite and match the finite difference of
//! the (verified) k-point analytic gradient.

use gfn1_rs::math::Vec3;
use gfn1_rs::pbc::gradient::pbc_analytic_gradient;
use gfn1_rs::pbc::hessian::pbc_kpoint_hessian;
use gfn1_rs::pbc::KMesh;
use gfn1_rs::{ElectronicOptions, Gfn1Parameters, PbcOptions, PeriodicSystem};

const BROMOETHANOL_CELL: &str = "9\nLattice=\"7.0 0 0 0 14 0 0 0 14\" pbc=\"T T T\"\n\
     C 0.000000 0.000000 0.000000\n\
     C 1.520000 0.000000 0.000000\n\
     O 2.160000 1.220000 0.000000\n\
     Br -1.940000 0.000000 0.000000\n\
     H 0.220000 1.020000 0.000000\n\
     H 0.220000 -0.510000 0.884000\n\
     H 1.880000 -0.510000 -0.884000\n\
     H 1.880000 -0.510000 0.884000\n\
     H 2.960000 1.100000 0.500000\n";

fn shift(system: &mut PeriodicSystem, dof: usize, delta: f64) {
    let atom = dof / 3;
    match dof % 3 {
        0 => system.atoms[atom].position.x += delta,
        1 => system.atoms[atom].position.y += delta,
        _ => system.atoms[atom].position.z += delta,
    }
}

fn component(v: Vec3, axis: usize) -> f64 {
    v.to_array()[axis]
}

#[test]
fn bromoethanol_kpoint_hessian_matches_gradient_fd() {
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    let base = PeriodicSystem::from_xyz_str(BROMOETHANOL_CELL, 0.0, false).unwrap();
    let opts = ElectronicOptions {
        energy_tolerance: 1.0e-10,
        charge_tolerance: 1.0e-9,
        max_scc: 500,
        ..ElectronicOptions::default()
    };
    let pbc = PbcOptions {
        kmesh: KMesh::monkhorst_pack([3, 1, 1]),
        ..PbcOptions::default()
    };

    let res = pbc_kpoint_hessian(&base, &params, &opts, &pbc).unwrap();
    let nat = base.atoms.len();
    let ndof = 3 * nat;

    // No unphysical blow-up.
    let mut maxabs = 0.0_f64;
    for i in 0..ndof {
        for j in 0..ndof {
            maxabs = maxabs.max(res.hessian[(i, j)].abs());
        }
    }
    assert!(maxabs < 1.0e3, "k-point Hessian blew up: {maxabs:.3e}");

    // Agreement with the finite difference of the k-point analytic gradient.
    let h = 1.0e-4;
    let grad = |system: &PeriodicSystem| {
        pbc_analytic_gradient(system, &params, &opts, &pbc)
            .unwrap()
            .gradient
    };
    let mut max_diff = 0.0_f64;
    for y in 0..ndof {
        let mut plus = base.clone();
        let mut minus = base.clone();
        shift(&mut plus, y, h);
        shift(&mut minus, y, -h);
        let gp = grad(&plus);
        let gm = grad(&minus);
        for atom in 0..nat {
            for axis in 0..3 {
                let fd = (component(gp[atom], axis) - component(gm[atom], axis)) / (2.0 * h);
                max_diff = max_diff.max((res.hessian[(3 * atom + axis, y)] - fd).abs());
            }
        }
    }
    println!("bromoethanol k-point Hessian vs gradient FD: max diff {max_diff:.3e}");
    assert!(
        max_diff < 1.0e-4,
        "k-point Hessian vs gradient FD max diff {max_diff:.3e}"
    );
}
