// SPDX-License-Identifier: GPL-3.0-or-later
//! Magnetic CP-SCC analytic magnetizability prototype (v0.1.6 foundation),
//! validated against the finite-field value. Variational 2nd derivative
//!   chi_aa = -[ Tr(P^a F^a) + Tr(P0 F^aa) - Tr(W^a S^a) - Tr(W0 S^aa) ],
//! F = H0 - V(.)S (V = converged SCC potential, fixed: dq/dB=0 at 1st order by
//! time reversal so the response is uncoupled). P^a, W^a are the first-order
//! density / energy-weighted-density responses from the F-eigenproblem; W^a
//! includes the first-order eigenvalue-response term eps_i^a C_i C_i^H.

use gfn1_rs::linalg::{lowdin_solve_generalized, Matrix};
use gfn1_rs::math::Vec3;
use gfn1_rs::pbc::complex::CMatrix;
use gfn1_rs::{
    magnetic_h0_overlap, run_magnetic_scc, ElectronicOptions, ExternalFieldOptions, Gfn1Parameters,
    PeriodicSystem, MAGNETIZABILITY_AU_TO_SI,
};

fn base_opts(b: Vec3) -> ElectronicOptions {
    ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1.0e-11,
        charge_tolerance: 1.0e-10,
        external_field: ExternalFieldOptions {
            magnetic_field: Some(b),
            origin: Vec3::zero(),
            ..ExternalFieldOptions::default()
        },
        ..ElectronicOptions::default()
    }
}

fn unit(axis: usize, s: f64) -> Vec3 {
    match axis {
        0 => Vec3::new(s, 0.0, 0.0),
        1 => Vec3::new(0.0, s, 0.0),
        _ => Vec3::new(0.0, 0.0, s),
    }
}

fn mo_transform(ct: &Matrix, c: &Matrix, m: &CMatrix) -> CMatrix {
    CMatrix {
        n: m.n,
        re: ct.matmul(&m.re).unwrap().matmul(c).unwrap(),
        im: ct.matmul(&m.im).unwrap().matmul(c).unwrap(),
    }
}

fn re_trace(a: &CMatrix, b: &CMatrix) -> f64 {
    let n = a.n;
    let mut acc = 0.0;
    for i in 0..n {
        for j in 0..n {
            acc += a.re[(i, j)] * b.re[(j, i)] - a.im[(i, j)] * b.im[(j, i)];
        }
    }
    acc
}

fn cmatmul(a: &CMatrix, b: &CMatrix) -> CMatrix {
    let n = a.n;
    let mut o = CMatrix::zeros(n);
    for i in 0..n {
        for k in 0..n {
            let (ar, ai) = (a.re[(i, k)], a.im[(i, k)]);
            if ar == 0.0 && ai == 0.0 {
                continue;
            }
            for j in 0..n {
                o.re[(i, j)] += ar * b.re[(k, j)] - ai * b.im[(k, j)];
                o.im[(i, j)] += ar * b.im[(k, j)] + ai * b.re[(k, j)];
            }
        }
    }
    o
}

/// Analytic magnetizability via McWeeny density-matrix CP-SCC (Malagoli 2010).
///   chi_aa = -[ Tr(P0 H0^aa) - Tr(W0 S^aa) + Tr(P^b H0^a) - Tr(W^b S^a) ]
/// The energy operator is H0 (band energy Tr(P H0)); the *response* P^b is driven
/// by the Fock derivative F^a = H0^a - V(.)S^a (V fixed: dq/dB=0 at first order by
/// time reversal, so the SCC kernel is uncoupled). The energy-weighted density
/// response uses the density-matrix identity W = 1/2 P F P (no occ-occ orbital
/// ambiguity): W^b = 1/2 [P^b F0 P0 + P0 F^b P0 + P0 F0 P^b], F^b = F^a.
fn analytic_chi(system: &PeriodicSystem, params: &Gfn1Parameters) -> [f64; 3] {
    let scc0 = run_magnetic_scc(system, params, &base_opts(Vec3::zero())).unwrap();
    let vao = &scc0.shell_potential_ao;
    let (h00, s00) = magnetic_h0_overlap(system, params, &base_opts(Vec3::zero()), None).unwrap();
    let n = h00.n;
    let mut f0 = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            f0[(i, j)] = h00.re[(i, j)] - 0.5 * (vao[i] + vao[j]) * s00.re[(i, j)];
        }
    }
    let eig = lowdin_solve_generalized(&f0, &s00.re, 1.0e-12).unwrap();
    let c = &eig.vectors;
    let eps = &eig.values;
    let ct = c.transpose();
    let nelec: f64 = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .map(|(i, j)| scc0.density.re[(i, j)] * s00.re[(j, i)])
        .sum();
    let nocc = (nelec / 2.0).round() as usize;
    let col = |m: &Matrix, k: usize| -> Vec<f64> { (0..n).map(|r| m[(r, k)]).collect() };
    let p0 = &scc0.density;
    let mut f0c = CMatrix::zeros(n);
    for i in 0..n {
        for j in 0..n {
            f0c.re[(i, j)] = f0[(i, j)];
        }
    }

    let d = 0.004;
    let mut chi = [0.0; 3];
    for axis in 0..3 {
        let (h0p, sp) =
            magnetic_h0_overlap(system, params, &base_opts(unit(axis, d)), None).unwrap();
        let (h0m, sm) =
            magnetic_h0_overlap(system, params, &base_opts(unit(axis, -d)), None).unwrap();
        let mut h0_a = CMatrix::zeros(n);
        let mut s_a = CMatrix::zeros(n);
        let mut f_a = CMatrix::zeros(n);
        let mut h0_aa = CMatrix::zeros(n);
        let mut s_aa = CMatrix::zeros(n);
        for i in 0..n {
            for j in 0..n {
                let v = 0.5 * (vao[i] + vao[j]);
                let dh0_re = (h0p.re[(i, j)] - h0m.re[(i, j)]) / (2.0 * d);
                let dh0_im = (h0p.im[(i, j)] - h0m.im[(i, j)]) / (2.0 * d);
                let ds_re = (sp.re[(i, j)] - sm.re[(i, j)]) / (2.0 * d);
                let ds_im = (sp.im[(i, j)] - sm.im[(i, j)]) / (2.0 * d);
                h0_a.re[(i, j)] = dh0_re;
                h0_a.im[(i, j)] = dh0_im;
                s_a.re[(i, j)] = ds_re;
                s_a.im[(i, j)] = ds_im;
                f_a.re[(i, j)] = dh0_re - v * ds_re;
                f_a.im[(i, j)] = dh0_im - v * ds_im;
                h0_aa.re[(i, j)] =
                    (h0p.re[(i, j)] - 2.0 * h00.re[(i, j)] + h0m.re[(i, j)]) / (d * d);
                h0_aa.im[(i, j)] =
                    (h0p.im[(i, j)] - 2.0 * h00.im[(i, j)] + h0m.im[(i, j)]) / (d * d);
                s_aa.re[(i, j)] = (sp.re[(i, j)] - 2.0 * s00.re[(i, j)] + sm.re[(i, j)]) / (d * d);
                s_aa.im[(i, j)] = (sp.im[(i, j)] - 2.0 * s00.im[(i, j)] + sm.im[(i, j)]) / (d * d);
            }
        }
        let fmo = mo_transform(&ct, c, &f_a);
        let smo = mo_transform(&ct, c, &s_a);
        let h0mo_aa = mo_transform(&ct, c, &h0_aa);
        let smo_aa = mo_transform(&ct, c, &s_aa);

        // Diamagnetic (no response): Tr(P0 H0^aa) - Tr(W0 S^aa), MO-diagonal.
        let mut dia = 0.0;
        for i in 0..nocc {
            dia += 2.0 * (h0mo_aa.re[(i, i)] - eps[i] * smo_aa.re[(i, i)]);
        }

        // Density response P^b: occ-occ = -1/2 S^a_mo (reorthonormalization),
        // occ-virt = canonical CPHF driven by F^a. Degeneracy-safe.
        let mut pa = CMatrix::zeros(n);
        for i in 0..nocc {
            let mut u_re = vec![0.0; n];
            let mut u_im = vec![0.0; n];
            for p in 0..n {
                if p < nocc {
                    u_re[p] = -0.5 * smo.re[(p, i)];
                    u_im[p] = -0.5 * smo.im[(p, i)];
                } else {
                    let denom = eps[i] - eps[p];
                    u_re[p] = (fmo.re[(p, i)] - eps[i] * smo.re[(p, i)]) / denom;
                    u_im[p] = (fmo.im[(p, i)] - eps[i] * smo.im[(p, i)]) / denom;
                }
            }
            let mut cia_re = vec![0.0; n];
            let mut cia_im = vec![0.0; n];
            for p in 0..n {
                let cp = col(c, p);
                for r in 0..n {
                    cia_re[r] += u_re[p] * cp[r];
                    cia_im[r] += u_im[p] * cp[r];
                }
            }
            let ci = col(c, i);
            for r in 0..n {
                for s in 0..n {
                    pa.re[(r, s)] += 2.0 * (cia_re[r] * ci[s] + ci[r] * cia_re[s]);
                    pa.im[(r, s)] += 2.0 * (cia_im[r] * ci[s] - ci[r] * cia_im[s]);
                }
            }
        }
        // Energy-weighted-density response W^b = 1/2 [P^b F0 P0 + P0 F^b P0 + P0 F0 P^b].
        let t1 = cmatmul(&cmatmul(&pa, &f0c), p0);
        let t2 = cmatmul(&cmatmul(p0, &f_a), p0);
        let t3 = cmatmul(&cmatmul(p0, &f0c), &pa);
        let mut wb = CMatrix::zeros(n);
        for i in 0..n {
            for j in 0..n {
                wb.re[(i, j)] = 0.5 * (t1.re[(i, j)] + t2.re[(i, j)] + t3.re[(i, j)]);
                wb.im[(i, j)] = 0.5 * (t1.im[(i, j)] + t2.im[(i, j)] + t3.im[(i, j)]);
            }
        }
        let para = re_trace(&pa, &h0_a) - re_trace(&wb, &s_a);
        // Charge-overlap response: B-derivative of -sum_mu vao_mu Re(P S^b)_mu,mu.
        // At B=0 the dvao/dB piece drops (dq/dB=0 and Re(P0 S^a)=0), leaving
        //   -sum_mu vao_mu [ Re(P^b S^a)_mu,mu + Re(P0 S^aa)_mu,mu ].
        let mut chargeov = 0.0;
        for mu in 0..n {
            let mut t = 0.0;
            for nu in 0..n {
                t += pa.re[(mu, nu)] * s_a.re[(nu, mu)] - pa.im[(mu, nu)] * s_a.im[(nu, mu)];
                t += p0.re[(mu, nu)] * s_aa.re[(nu, mu)] - p0.im[(mu, nu)] * s_aa.im[(nu, mu)];
            }
            chargeov -= vao[mu] * t;
        }
        chi[axis] = -(dia + para + chargeov) * MAGNETIZABILITY_AU_TO_SI;
    }
    chi
}

/// Diagnostic: chi from the analytic diamagnetic (no-response) part PLUS the
/// response evaluated with a FINITE-DIFFERENCE of the converged SCC density /
/// energy-weighted density (dP/dB, dW/dB). Localizes whether the analytic response
/// (P^b/W^b) or the diamagnetic/contraction is the source of the gap to FD.
fn hybrid_chi(system: &PeriodicSystem, params: &Gfn1Parameters) -> [f64; 3] {
    let scc0 = run_magnetic_scc(system, params, &base_opts(Vec3::zero())).unwrap();
    let vao = &scc0.shell_potential_ao;
    let (h00, s00) = magnetic_h0_overlap(system, params, &base_opts(Vec3::zero()), None).unwrap();
    let n = h00.n;
    let mut f0 = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            f0[(i, j)] = h00.re[(i, j)] - 0.5 * (vao[i] + vao[j]) * s00.re[(i, j)];
        }
    }
    let eig = lowdin_solve_generalized(&f0, &s00.re, 1.0e-12).unwrap();
    let c = &eig.vectors;
    let eps = &eig.values;
    let ct = c.transpose();
    let nelec: f64 = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .map(|(i, j)| scc0.density.re[(i, j)] * s00.re[(j, i)])
        .sum();
    let nocc = (nelec / 2.0).round() as usize;
    let d = 0.004;
    let mut chi = [0.0; 3];
    for axis in 0..3 {
        let (h0p, sp) =
            magnetic_h0_overlap(system, params, &base_opts(unit(axis, d)), None).unwrap();
        let (h0m, sm) =
            magnetic_h0_overlap(system, params, &base_opts(unit(axis, -d)), None).unwrap();
        // H0^a, S^a (imaginary), H0^aa, S^aa (real).
        let mut h0_a = CMatrix::zeros(n);
        let mut s_a = CMatrix::zeros(n);
        let mut h0_aa = CMatrix::zeros(n);
        let mut s_aa = CMatrix::zeros(n);
        for i in 0..n {
            for j in 0..n {
                h0_a.re[(i, j)] = (h0p.re[(i, j)] - h0m.re[(i, j)]) / (2.0 * d);
                h0_a.im[(i, j)] = (h0p.im[(i, j)] - h0m.im[(i, j)]) / (2.0 * d);
                s_a.re[(i, j)] = (sp.re[(i, j)] - sm.re[(i, j)]) / (2.0 * d);
                s_a.im[(i, j)] = (sp.im[(i, j)] - sm.im[(i, j)]) / (2.0 * d);
                h0_aa.re[(i, j)] =
                    (h0p.re[(i, j)] - 2.0 * h00.re[(i, j)] + h0m.re[(i, j)]) / (d * d);
                s_aa.re[(i, j)] = (sp.re[(i, j)] - 2.0 * s00.re[(i, j)] + sm.re[(i, j)]) / (d * d);
            }
        }
        // Analytic diamagnetic (no response), using H0^aa (NOT F^aa).
        let h0mo_aa = mo_transform(&ct, c, &h0_aa);
        let smo_aa = mo_transform(&ct, c, &s_aa);
        let mut dia = 0.0;
        for i in 0..nocc {
            dia += 2.0 * (h0mo_aa.re[(i, i)] - eps[i] * smo_aa.re[(i, i)]);
        }
        // Response from FD of the converged SCC density / EW density.
        let pp = run_magnetic_scc(system, params, &base_opts(unit(axis, d))).unwrap();
        let pm = run_magnetic_scc(system, params, &base_opts(unit(axis, -d))).unwrap();
        let mut dp = CMatrix::zeros(n);
        let mut dw = CMatrix::zeros(n);
        for i in 0..n {
            for j in 0..n {
                dp.re[(i, j)] = (pp.density.re[(i, j)] - pm.density.re[(i, j)]) / (2.0 * d);
                dp.im[(i, j)] = (pp.density.im[(i, j)] - pm.density.im[(i, j)]) / (2.0 * d);
                dw.re[(i, j)] = (pp.energy_weighted_density.re[(i, j)]
                    - pm.energy_weighted_density.re[(i, j)])
                    / (2.0 * d);
                dw.im[(i, j)] = (pp.energy_weighted_density.im[(i, j)]
                    - pm.energy_weighted_density.im[(i, j)])
                    / (2.0 * d);
            }
        }
        let resp = re_trace(&dp, &h0_a) - re_trace(&dw, &s_a);
        chi[axis] = -(dia + resp) * MAGNETIZABILITY_AU_TO_SI;
    }
    chi
}

/// Compare the analytic P^b / W^b response to the FD-of-SCC-density response for one
/// molecule (axis z), to localize the bug to P^b or W^b.
fn debug_response(name: &str, system: &PeriodicSystem, params: &Gfn1Parameters) {
    let axis = 2usize;
    let scc0 = run_magnetic_scc(system, params, &base_opts(Vec3::zero())).unwrap();
    let vao = &scc0.shell_potential_ao;
    let (h00, s00) = magnetic_h0_overlap(system, params, &base_opts(Vec3::zero()), None).unwrap();
    let n = h00.n;
    let mut f0 = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            f0[(i, j)] = h00.re[(i, j)] - 0.5 * (vao[i] + vao[j]) * s00.re[(i, j)];
        }
    }
    let eig = lowdin_solve_generalized(&f0, &s00.re, 1.0e-12).unwrap();
    let c = &eig.vectors;
    let eps = &eig.values;
    let ct = c.transpose();
    let nelec: f64 = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .map(|(i, j)| scc0.density.re[(i, j)] * s00.re[(j, i)])
        .sum();
    let nocc = (nelec / 2.0).round() as usize;
    let col = |m: &Matrix, k: usize| -> Vec<f64> { (0..n).map(|r| m[(r, k)]).collect() };
    let d = 0.004;
    let (h0p, sp) = magnetic_h0_overlap(system, params, &base_opts(unit(axis, d)), None).unwrap();
    let (h0m, sm) = magnetic_h0_overlap(system, params, &base_opts(unit(axis, -d)), None).unwrap();
    let mut h0_a = CMatrix::zeros(n);
    let mut s_a = CMatrix::zeros(n);
    for i in 0..n {
        for j in 0..n {
            h0_a.re[(i, j)] = (h0p.re[(i, j)] - h0m.re[(i, j)]) / (2.0 * d);
            h0_a.im[(i, j)] = (h0p.im[(i, j)] - h0m.im[(i, j)]) / (2.0 * d);
            s_a.re[(i, j)] = (sp.re[(i, j)] - sm.re[(i, j)]) / (2.0 * d);
            s_a.im[(i, j)] = (sp.im[(i, j)] - sm.im[(i, j)]) / (2.0 * d);
        }
    }
    // Fock derivative F^a = H0^a - V(.)S^a (V fixed: uncoupled magnetic response).
    let mut f_a = CMatrix::zeros(n);
    for i in 0..n {
        for j in 0..n {
            let v = 0.5 * (vao[i] + vao[j]);
            f_a.re[(i, j)] = h0_a.re[(i, j)] - v * s_a.re[(i, j)];
            f_a.im[(i, j)] = h0_a.im[(i, j)] - v * s_a.im[(i, j)];
        }
    }
    let fmo = mo_transform(&ct, c, &f_a); // Fock-derivative RHS
    let smo = mo_transform(&ct, c, &s_a);
    // Density response P^b: occ-occ block = -1/2 S^a_mo (reorthonormalization),
    // occ-virt block = canonical CPHF. Invariant to degenerate-set rotations.
    let mut pa = CMatrix::zeros(n);
    let mut wa = CMatrix::zeros(n);
    for i in 0..nocc {
        let mut u_re = vec![0.0; n];
        let mut u_im = vec![0.0; n];
        for p in 0..n {
            if p < nocc {
                u_re[p] = -0.5 * smo.re[(p, i)];
                u_im[p] = -0.5 * smo.im[(p, i)];
            } else {
                let denom = eps[i] - eps[p];
                u_re[p] = (fmo.re[(p, i)] - eps[i] * smo.re[(p, i)]) / denom;
                u_im[p] = (fmo.im[(p, i)] - eps[i] * smo.im[(p, i)]) / denom;
            }
        }
        let mut cre = vec![0.0; n];
        let mut cim = vec![0.0; n];
        for p in 0..n {
            let cp = col(c, p);
            for r in 0..n {
                cre[r] += u_re[p] * cp[r];
                cim[r] += u_im[p] * cp[r];
            }
        }
        let ci = col(c, i);
        for r in 0..n {
            for s in 0..n {
                let pre = 2.0 * (cre[r] * ci[s] + ci[r] * cre[s]);
                let pim = 2.0 * (cim[r] * ci[s] - ci[r] * cim[s]);
                pa.re[(r, s)] += pre;
                pa.im[(r, s)] += pim;
                wa.re[(r, s)] += eps[i] * pre;
                wa.im[(r, s)] += eps[i] * pim;
            }
        }
    }
    // FD response.
    let pp = run_magnetic_scc(system, params, &base_opts(unit(axis, d))).unwrap();
    let pm = run_magnetic_scc(system, params, &base_opts(unit(axis, -d))).unwrap();
    let mut dp = CMatrix::zeros(n);
    let mut dw = CMatrix::zeros(n);
    for i in 0..n {
        for j in 0..n {
            dp.re[(i, j)] = (pp.density.re[(i, j)] - pm.density.re[(i, j)]) / (2.0 * d);
            dp.im[(i, j)] = (pp.density.im[(i, j)] - pm.density.im[(i, j)]) / (2.0 * d);
            dw.re[(i, j)] = (pp.energy_weighted_density.re[(i, j)]
                - pm.energy_weighted_density.re[(i, j)])
                / (2.0 * d);
            dw.im[(i, j)] = (pp.energy_weighted_density.im[(i, j)]
                - pm.energy_weighted_density.im[(i, j)])
                / (2.0 * d);
        }
    }
    // Density-matrix W^b = 1/2 [P^b F0 P0 + P0 F^b P0 + P0 F0 P^b], F = H0 - V(.)S
    // (W = D F D, D = 1/2 P). No occ-occ orbital ambiguity.
    let cmm = |a: &CMatrix, b: &CMatrix| -> CMatrix {
        let mut o = CMatrix::zeros(n);
        for i in 0..n {
            for k in 0..n {
                let (ar, ai) = (a.re[(i, k)], a.im[(i, k)]);
                if ar == 0.0 && ai == 0.0 {
                    continue;
                }
                for j in 0..n {
                    o.re[(i, j)] += ar * b.re[(k, j)] - ai * b.im[(k, j)];
                    o.im[(i, j)] += ar * b.im[(k, j)] + ai * b.re[(k, j)];
                }
            }
        }
        o
    };
    let p0 = &scc0.density;
    let mut f0c = CMatrix::zeros(n);
    let mut fbc = CMatrix::zeros(n);
    for i in 0..n {
        for j in 0..n {
            let v = 0.5 * (vao[i] + vao[j]);
            f0c.re[(i, j)] = f0[(i, j)];
            fbc.re[(i, j)] = h0_a.re[(i, j)] - v * s_a.re[(i, j)];
            fbc.im[(i, j)] = h0_a.im[(i, j)] - v * s_a.im[(i, j)];
        }
    }
    let t1 = cmm(&cmm(&pa, &f0c), p0);
    let t2 = cmm(&cmm(p0, &fbc), p0);
    let t3 = cmm(&cmm(p0, &f0c), &pa);
    let mut wb_dm = CMatrix::zeros(n);
    for i in 0..n {
        for j in 0..n {
            wb_dm.re[(i, j)] = 0.5 * (t1.re[(i, j)] + t2.re[(i, j)] + t3.re[(i, j)]);
            wb_dm.im[(i, j)] = 0.5 * (t1.im[(i, j)] + t2.im[(i, j)] + t3.im[(i, j)]);
        }
    }
    // Verify W0 = 1/2 P0 F0 P0 (density-matrix energy-weighted density identity).
    let w0_dm = cmm(&cmm(p0, &f0c), p0);
    let mut w0_err = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let d = 0.5 * w0_dm.re[(i, j)] - scc0.energy_weighted_density.re[(i, j)];
            w0_err = w0_err.max(d.abs());
        }
    }
    println!(
        "  {name} zz:  Tr(P^b H0^a) ana={:.4} fd={:.4} | Tr(W^b S^a) orb={:.4} dm={:.4} fd={:.4} | W0-id err={:.1e}",
        re_trace(&pa, &h0_a), re_trace(&dp, &h0_a), re_trace(&wa, &s_a), re_trace(&wb_dm, &s_a), re_trace(&dw, &s_a), w0_err
    );
}

fn fd_chi_step(system: &PeriodicSystem, params: &Gfn1Parameters, d: f64) -> [f64; 3] {
    let e0 = run_magnetic_scc(system, params, &base_opts(Vec3::zero()))
        .unwrap()
        .energy;
    let mut chi = [0.0; 3];
    for axis in 0..3 {
        let ep = run_magnetic_scc(system, params, &base_opts(unit(axis, d)))
            .unwrap()
            .energy;
        let em = run_magnetic_scc(system, params, &base_opts(unit(axis, -d)))
            .unwrap()
            .energy;
        chi[axis] = -((ep - 2.0 * e0 + em) / (d * d)) * MAGNETIZABILITY_AU_TO_SI;
    }
    chi
}

fn fd_chi(system: &PeriodicSystem, params: &Gfn1Parameters) -> [f64; 3] {
    // Richardson extrapolation d->0 from the central 2nd-difference (removes the
    // O(d^2) hyper-magnetizability/B^4 contamination): chi = (4*chi(d/2)-chi(d))/3.
    let c1 = fd_chi_step(system, params, 0.008);
    let c2 = fd_chi_step(system, params, 0.004);
    let mut chi = [0.0; 3];
    for a in 0..3 {
        chi[a] = (4.0 * c2[a] - c1[a]) / 3.0;
    }
    chi
}

/// Decisive: test the 1st-derivative Hellmann-Feynman formula
///   dE/dB = Re[ Tr(P H0^b) - Tr(W S^b) ]
/// at a FINITE field h (so it is nonzero), against the FD of the energy. If these
/// match, the energy/density/W are all consistent and the 2nd derivative must too;
/// if not, the 1st-derivative formula is missing a term.
fn fd1_check(name: &str, system: &PeriodicSystem, params: &Gfn1Parameters) {
    let axis = 2usize;
    let h = 0.02;
    let scc_h = run_magnetic_scc(system, params, &base_opts(unit(axis, h))).unwrap();
    let n = scc_h.density.n;
    for delta in [0.004, 0.001] {
        let (h0p, sp) =
            magnetic_h0_overlap(system, params, &base_opts(unit(axis, h + delta)), None).unwrap();
        let (h0m, sm) =
            magnetic_h0_overlap(system, params, &base_opts(unit(axis, h - delta)), None).unwrap();
        let mut h0b = CMatrix::zeros(n);
        let mut sb = CMatrix::zeros(n);
        for i in 0..n {
            for j in 0..n {
                h0b.re[(i, j)] = (h0p.re[(i, j)] - h0m.re[(i, j)]) / (2.0 * delta);
                h0b.im[(i, j)] = (h0p.im[(i, j)] - h0m.im[(i, j)]) / (2.0 * delta);
                sb.re[(i, j)] = (sp.re[(i, j)] - sm.re[(i, j)]) / (2.0 * delta);
                sb.im[(i, j)] = (sp.im[(i, j)] - sm.im[(i, j)]) / (2.0 * delta);
            }
        }
        // Explicit-overlap Mulliken term: q = ref - Re(PS) depends on B via S(B), so
        // d(E_scc)/dB picks up -sum_mu vao_mu Re(P S^b)_mu,mu (per-AO, not symmetric:
        // the symmetric (vao_i+vao_j)/2 form vanishes for anti-Hermitian S^b).
        let p_h = &scc_h.density;
        let vao_h = &scc_h.shell_potential_ao;
        let mut charge_ov = 0.0;
        for mu in 0..n {
            let mut ps = 0.0;
            for nu in 0..n {
                ps += p_h.re[(mu, nu)] * sb.re[(nu, mu)] - p_h.im[(mu, nu)] * sb.im[(nu, mu)];
            }
            charge_ov += vao_h[mu] * ps;
        }
        let de_ana = re_trace(p_h, &h0b) - re_trace(&scc_h.energy_weighted_density, &sb);
        let ep = run_magnetic_scc(system, params, &base_opts(unit(axis, h + delta)))
            .unwrap()
            .energy;
        let em = run_magnetic_scc(system, params, &base_opts(unit(axis, h - delta)))
            .unwrap()
            .energy;
        let de_fd = (ep - em) / (2.0 * delta);
        println!(
            "  {name} dE/dB(h={h},d={delta}):  HF={:.6} HF-chargeov={:.6} fd={:.6}",
            de_ana,
            de_ana - charge_ov,
            de_fd
        );
    }
}

fn main() {
    let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
    let mols = [
        ("HF", "2\nHF\nH 0.0 0.0 0.0\nF 0.9168 0.0 0.0\n"),
        ("H2O", "3\nwater\nO 0.0 0.0 0.0\nH 0.7572 0.5864 0.0\nH -0.7572 0.5864 0.0\n"),
        ("CH4", "5\nCH4\nC 0.0 0.0 0.0\nH 0.6276 0.6276 0.6276\nH 0.6276 -0.6276 -0.6276\nH -0.6276 0.6276 -0.6276\nH -0.6276 -0.6276 0.6276\n"),
    ];
    println!("Analytic CP-SCC vs finite-field magnetizability (1e-30 J/T^2):");
    for (name, xyz) in mols {
        let system = PeriodicSystem::from_xyz_str(xyz, 0.0, false).unwrap();
        let a = analytic_chi(&system, &params);
        let hy = hybrid_chi(&system, &params);
        let f = fd_chi(&system, &params);
        let ia = (a[0] + a[1] + a[2]) / 3.0;
        let ih = (hy[0] + hy[1] + hy[2]) / 3.0;
        let ifd = (f[0] + f[1] + f[2]) / 3.0;
        println!(
            "{name:5} analytic {:8.2} | hybrid(dia+FDresp) {:8.2} | FD {:8.2}",
            ia, ih, ifd
        );
        debug_response(name, &system, &params);
        fd1_check(name, &system, &params);
    }
}
