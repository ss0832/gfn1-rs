// SPDX-License-Identifier: GPL-3.0-or-later
//! Independent verification of the complex London (GIAO) kinetic integral
//! `<omega_a|1/2 pi^2|omega_b>` by brute-force real-space integration on a grid,
//! using *analytic* orbital derivatives (so only the spatial quadrature is
//! approximate). Compares against `lao_kinetic_matrix` for a two-center p-orbital
//! matrix element under a perpendicular field. This isolates whether the
//! paramagnetic / diamagnetic kinetic terms are correct.

use gfn1_rs::basis::{AOBasisFunction, BasisOptions, BasisSet};
use gfn1_rs::magnetic::lao_kinetic_matrix;
use gfn1_rs::math::Vec3;
use gfn1_rs::{ExternalFieldOptions, Gfn1Parameters, PeriodicSystem};

/// Evaluate a contracted real AO and its gradient + Laplacian at point `r`.
/// Returns (phi, grad_phi, lap_phi).
fn eval_ao(ao: &AOBasisFunction, center: Vec3, r: Vec3) -> (f64, Vec3, f64) {
    let s = r - center;
    // Radial contraction R(s) = sum_p c_p exp(-a_p |s|^2) and its s-derivatives.
    let r2 = s.dot(s);
    let mut rad = 0.0;
    let mut rad_da = 0.0; // sum_p c_p (-2 a_p) exp   (so dR/dx = rad_da * s_x)
    let mut rad_d2 = 0.0; // sum_p c_p (4 a_p^2 |s|^2 - 6 a_p) exp  (lap of radial)
    for p in &ao.primitives {
        let e = (-p.exponent * r2).exp() * p.coefficient;
        rad += e;
        rad_da += -2.0 * p.exponent * e;
        rad_d2 += (4.0 * p.exponent * p.exponent * r2 - 6.0 * p.exponent) * e;
    }
    // Angular polynomial P(s) = sum_c c_c s_x^px s_y^py s_z^pz and derivatives.
    let pow = |v: f64, n: i32| -> f64 {
        if n < 0 {
            0.0
        } else {
            v.powi(n)
        }
    };
    let mut ang = 0.0;
    let mut grad_ang = Vec3::zero();
    let mut lap_ang = 0.0;
    for c in &ao.components {
        let (px, py, pz) = (c.power.x as i32, c.power.y as i32, c.power.z as i32);
        let base = pow(s.x, px) * pow(s.y, py) * pow(s.z, pz);
        ang += c.coefficient * base;
        // gradient of the monomial
        let gx = px as f64 * pow(s.x, px - 1) * pow(s.y, py) * pow(s.z, pz);
        let gy = py as f64 * pow(s.x, px) * pow(s.y, py - 1) * pow(s.z, pz);
        let gz = pz as f64 * pow(s.x, px) * pow(s.y, py) * pow(s.z, pz - 1);
        grad_ang.x += c.coefficient * gx;
        grad_ang.y += c.coefficient * gy;
        grad_ang.z += c.coefficient * gz;
        // laplacian of the monomial
        let lx = (px * (px - 1)) as f64 * pow(s.x, px - 2) * pow(s.y, py) * pow(s.z, pz);
        let ly = (py * (py - 1)) as f64 * pow(s.x, px) * pow(s.y, py - 2) * pow(s.z, pz);
        let lz = (pz * (pz - 1)) as f64 * pow(s.x, px) * pow(s.y, py) * pow(s.z, pz - 2);
        lap_ang += c.coefficient * (lx + ly + lz);
    }
    // phi = ang * rad
    let phi = ang * rad;
    // grad phi = (grad_ang) rad + ang (grad rad);  grad rad = rad_da * s
    let grad = Vec3::new(
        grad_ang.x * rad + ang * rad_da * s.x,
        grad_ang.y * rad + ang * rad_da * s.y,
        grad_ang.z * rad + ang * rad_da * s.z,
    );
    // lap phi = (lap_ang) rad + 2 grad_ang . grad_rad + ang (lap rad)
    //   grad_rad = rad_da * s ;  lap_rad = rad_d2
    let lap = lap_ang * rad + 2.0 * (grad_ang.dot(s)) * rad_da + ang * rad_d2;
    (phi, grad, lap)
}

fn main() {
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    // Water; O has 2s,2px,2py,2pz (AOs 0..4), H1 1s = AO 4, H2 1s = AO 5.
    let system = PeriodicSystem::from_xyz_str(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n",
        0.0,
        false,
    )
    .unwrap();
    let basis = BasisSet::build(&system, &params, BasisOptions::default()).unwrap();
    let centers: Vec<Vec3> = basis
        .aos
        .iter()
        .map(|a| system.atoms[a.atom_index].position)
        .collect();

    let bfield = Vec3::new(0.0, 0.0, 0.06); // perpendicular to the molecular plane
    let origin = Vec3::new(0.2, -0.1, 0.05); // off-center gauge origin
    let opts = ExternalFieldOptions {
        magnetic_field: Some(bfield),
        origin,
        ..Default::default()
    };
    let k_analytic = lao_kinetic_matrix(&system, &basis, &opts);

    // Grid.
    let lo = -8.0_f64;
    let hi = 8.0_f64;
    let ngrid = 220usize;
    let dx = (hi - lo) / ngrid as f64;
    let dv = dx * dx * dx;

    // Check a few representative elements (s-s, p-s two-center, p-p).
    let pairs = [(0usize, 4usize), (1, 4), (1, 1), (1, 2), (0, 1)];
    println!(
        "LAO kinetic <om_a|1/2 pi^2|om_b>: grid vs analytic (B={:?})",
        bfield
    );
    for (a, b) in pairs {
        let ka = bfield.cross(centers[a] - origin) * 0.5;
        let kb = bfield.cross(centers[b] - origin) * 0.5;
        let (mut re, mut im) = (0.0_f64, 0.0_f64);
        for ix in 0..ngrid {
            let x = lo + (ix as f64 + 0.5) * dx;
            for iy in 0..ngrid {
                let y = lo + (iy as f64 + 0.5) * dx;
                for iz in 0..ngrid {
                    let z = lo + (iz as f64 + 0.5) * dx;
                    let r = Vec3::new(x, y, z);
                    let (phi_a, _, _) = eval_ao(&basis.aos[a], centers[a], r);
                    if phi_a.abs() < 1.0e-300 {
                        continue;
                    }
                    let (phi_b, grad_b, lap_b) = eval_ao(&basis.aos[b], centers[b], r);
                    let avec = bfield.cross(r - origin) * 0.5;
                    // 1/2 pi^2 phi_b (the bracket multiplying e^{-i k_b r}):
                    //   real:  -1/2 lap_b + 1/2 |k_b|^2 phi_b - (A.k_b) phi_b + 1/2 |A|^2 phi_b
                    //   imag:  (k_b - A) . grad_b
                    let real_op = -0.5 * lap_b + 0.5 * kb.dot(kb) * phi_b - avec.dot(kb) * phi_b
                        + 0.5 * avec.dot(avec) * phi_b;
                    let imag_op = (kb - avec).dot(grad_b);
                    // multiply by phi_a e^{+i k_a r} e^{-i k_b r} = phi_a e^{i (k_a-k_b) r}
                    let q = ka - kb;
                    let theta = q.dot(r);
                    let (ct, st) = (theta.cos(), theta.sin());
                    // (real_op + i imag_op) * (ct + i st), then * phi_a
                    re += phi_a * (real_op * ct - imag_op * st);
                    im += phi_a * (real_op * st + imag_op * ct);
                }
            }
        }
        re *= dv;
        im *= dv;
        let ar = k_analytic.re[(a, b)];
        let ai = k_analytic.im[(a, b)];
        println!(
            "  ({a},{b}) [{:>3} {:>3}]: grid=({:+.5},{:+.5}) analytic=({:+.5},{:+.5}) d=({:.1e},{:.1e})",
            basis.aos[a].cart_label,
            basis.aos[b].cart_label,
            re, im, ar, ai, (re - ar).abs(), (im - ai).abs()
        );
    }
}
