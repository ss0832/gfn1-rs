// SPDX-License-Identifier: GPL-3.0-or-later
//! Larger complex pencils exercise the backend beyond small-matrix dispatch.
use gfn1_rs::pbc::complex::{native_generalized_eigen, CMatrix};
use serde_json::json;
use sha2::{Digest, Sha256};

fn value(i: usize, j: usize) -> f64 {
    ((i * 17 + j * 13 + 11) % 97) as f64 / 97. - 0.5
}

#[test]
fn complex_generalized_eigenpairs_match_independent_scalar_residuals() {
    let mut cases = Vec::new();
    for n in [33, 65, 129] {
        let mut a = CMatrix::zeros(n);
        let mut h = CMatrix::zeros(n);
        for i in 0..n {
            for j in 0..n {
                a.re[(i, j)] = value(i, j);
                a.im[(i, j)] = 0.3 * value(i + 3, j + 5);
                h.re[(i, j)] = 0.5 * (value(i, j) + value(j, i));
                h.im[(i, j)] = 0.2 * (value(i, j) - value(j, i));
            }
        }
        let mut s = a.adjoint().matmul(&a).unwrap();
        for i in 0..n {
            for j in 0..n {
                s.re[(i, j)] *= 0.01 / n as f64;
                s.im[(i, j)] *= 0.01 / n as f64;
            }
            s.re[(i, i)] += 1.;
        }
        let eigen = native_generalized_eigen(&h, &s, 1e-12).unwrap();
        let c = &eigen.vectors;
        let mut hc = CMatrix::zeros(n);
        let mut sc = CMatrix::zeros(n);
        // Deliberately scalar: do not validate GEMM with another GEMM call.
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    hc.re[(i, j)] += h.re[(i, k)] * c.re[(k, j)] - h.im[(i, k)] * c.im[(k, j)];
                    hc.im[(i, j)] += h.re[(i, k)] * c.im[(k, j)] + h.im[(i, k)] * c.re[(k, j)];
                    sc.re[(i, j)] += s.re[(i, k)] * c.re[(k, j)] - s.im[(i, k)] * c.im[(k, j)];
                    sc.im[(i, j)] += s.re[(i, k)] * c.im[(k, j)] + s.im[(i, k)] * c.re[(k, j)];
                }
            }
        }
        let mut residual = 0_f64;
        let mut orthogonal = 0_f64;
        for i in 0..n {
            for j in 0..n {
                residual = residual.max((hc.re[(i, j)] - sc.re[(i, j)] * eigen.values[j]).abs());
                residual = residual.max((hc.im[(i, j)] - sc.im[(i, j)] * eigen.values[j]).abs());
                let mut re = 0.;
                let mut im = 0.;
                for k in 0..n {
                    re += c.re[(k, i)] * sc.re[(k, j)] + c.im[(k, i)] * sc.im[(k, j)];
                    im += c.re[(k, i)] * sc.im[(k, j)] - c.im[(k, i)] * sc.re[(k, j)];
                }
                orthogonal = orthogonal
                    .max((re - if i == j { 1. } else { 0. }).abs())
                    .max(im.abs());
            }
        }
        assert!(
            residual < 1e-10 && orthogonal < 1e-11,
            "n={n}: H C - S C eps={residual:e}, C^H S C - I={orthogonal:e}"
        );
        cases.push(json!({"dimension":n,"max_eigenpair_residual":residual,"max_overlap_orthogonality_residual":orthogonal}));
    }
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/validation/gemm-update");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("complex-linalg.json"),serde_json::to_vec_pretty(&json!({"status":"pass","package_version":env!("CARGO_PKG_VERSION"),"dependency_lockfile_sha256":format!("{:x}",Sha256::digest(include_bytes!("../Cargo.lock"))),"cases":cases})).unwrap()).unwrap();
}
