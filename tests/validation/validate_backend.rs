// SPDX-License-Identifier: GPL-3.0-or-later
//! Developer-only validation/diagnostic driver; see tests/README.md.
//! Deterministic numerical/latency audit across dense-backend lockfile updates.
//! Build once with each lockfile, then run the two preserved binaries alternately.
use faer::linalg::matmul::triangular::{matmul, BlockStructure};
use faer::{Accum, MatMut, MatRef, Par};
use gfn1_rs::linalg::{symmetric_eigen, Matrix};
use gfn1_rs::*;
use serde_json::{json, Value};
use sha2::Digest;
use std::time::Instant;

fn sample(i: usize, j: usize, salt: u64) -> f64 {
    let mut x = (i as u64 + 1).wrapping_mul(6364136223846793005)
        ^ (j as u64 + salt).wrapping_mul(1442695040888963407);
    x ^= x >> 29;
    (x >> 11) as f64 / ((1u64 << 53) as f64) - 0.5
}
fn latency(mut f: impl FnMut()) -> Value {
    for _ in 0..5 {
        f();
    }
    let start = Instant::now();
    let mut pilot_calls = 0;
    while start.elapsed().as_secs_f64() < 0.005 {
        f();
        pilot_calls += 1;
    }
    let per_call = start.elapsed().as_secs_f64() / pilot_calls as f64;
    let repetitions = (0.020 / per_call.max(1e-9)).ceil().clamp(1., 1_000_000.) as usize;
    let mut samples = Vec::new();
    for _ in 0..7 {
        let start = Instant::now();
        for _ in 0..repetitions {
            f();
        }
        samples.push(start.elapsed().as_secs_f64() / repetitions as f64);
    }
    samples.sort_by(f64::total_cmp);
    json!({"median_seconds":samples[3],"samples_seconds":samples,"calls_per_sample":repetitions})
}
fn kernel(
    m: usize,
    n: usize,
    k: usize,
    row_major: bool,
    structure: BlockStructure,
    parallel: bool,
) -> Value {
    let index =
        |i: usize, j: usize, r: usize, c: usize| if row_major { i * c + j } else { i + j * r };
    let mut a = vec![0.; m * k];
    let mut b = vec![0.; k * n];
    for i in 0..m {
        for j in 0..k {
            a[index(i, j, m, k)] = sample(i, j, 7);
        }
    }
    for i in 0..k {
        for j in 0..n {
            b[index(i, j, k, n)] = sample(i, j, 19);
        }
    }
    let mut out = vec![f64::NAN; m * n];
    let mut calculate = || {
        let (dst, lhs, rhs) = if row_major {
            (
                MatMut::from_row_major_slice_mut(&mut out, m, n),
                MatRef::from_row_major_slice(&a, m, k),
                MatRef::from_row_major_slice(&b, k, n),
            )
        } else {
            (
                MatMut::from_column_major_slice_mut(&mut out, m, n),
                MatRef::from_column_major_slice(&a, m, k),
                MatRef::from_column_major_slice(&b, k, n),
            )
        };
        matmul(
            dst,
            structure,
            Accum::Replace,
            lhs,
            BlockStructure::Rectangular,
            rhs,
            BlockStructure::Rectangular,
            1.,
            if parallel { Par::rayon(2) } else { Par::Seq },
        );
    };
    let timing = latency(&mut calculate);
    let mut max_error = 0_f64;
    let mut max_ratio = 0_f64;
    let mut tested = 0;
    let mut nonfinite = 0;
    let mut outside_overwrites = 0;
    for i in 0..m {
        for j in 0..n {
            let included = match structure {
                BlockStructure::TriangularLower => i >= j,
                BlockStructure::TriangularUpper => i <= j,
                _ => true,
            };
            let value = out[index(i, j, m, n)];
            if !included {
                outside_overwrites += usize::from(!value.is_nan());
                continue;
            }
            let mut sum = 0.;
            let mut compensation = 0.;
            let mut absolute = 0.;
            for r in 0..k {
                let product = a[index(i, r, m, k)] * b[index(r, j, k, n)];
                absolute += product.abs();
                let y = product - compensation;
                let next = sum + y;
                compensation = (next - sum) - y;
                sum = next;
            }
            if !value.is_finite() {
                nonfinite += 1;
                tested += 1;
                continue;
            }
            let error = (value - sum).abs();
            let tolerance = 16. * f64::EPSILON * k as f64 * absolute.max(1.);
            max_error = max_error.max(error);
            max_ratio = max_ratio.max(error / tolerance);
            tested += 1;
        }
    }
    json!({"kind":"gemm","m":m,"n":n,"k":k,"row_major":row_major,"structure":format!("{structure:?}"),
        "status":if max_ratio<=1. && nonfinite==0 && outside_overwrites==0 {"pass"}else{"fail"},
        "included_nonfinite":nonfinite,"excluded_triangle_overwrites":outside_overwrites,
        "parallel_threads":if parallel {2}else{1},"tested_components":tested,"max_abs_error":max_error,
        "max_error_over_roundoff_bound":max_ratio,"reference":"independent compensated scalar dot products; excluded triangle remains NaN", "timing":timing})
}
fn main() -> Result<()> {
    let output = std::env::args().nth(1).expect("output JSON path");
    let mut cases = Vec::new();
    for (m, n, k) in [
        (17, 19, 23),
        (64, 65, 63),
        (127, 129, 65),
        (257, 259, 131),
        (513, 509, 257),
    ] {
        for row in [true, false] {
            for parallel in [false, true] {
                cases.push(kernel(m, n, k, row, BlockStructure::Rectangular, parallel));
            }
        }
    }
    for n in [33, 129, 257, 513] {
        for row in [true, false] {
            for parallel in [false, true] {
                for part in [
                    BlockStructure::TriangularLower,
                    BlockStructure::TriangularUpper,
                ] {
                    cases.push(kernel(n, n, 131, row, part, parallel));
                }
            }
        }
    }
    let n = 129;
    let a = Matrix::from_vec(n, n, (0..n * n).map(|x| sample(x / n, x % n, 31)).collect())?;
    let mut spd = a.matmul(&a.transpose())?;
    for i in 0..n {
        spd[(i, i)] += 1.;
    }
    let eig = symmetric_eigen(&spd)?;
    let av = spd.matmul(&eig.vectors)?;
    let mut residual = 0_f64;
    for i in 0..n {
        for j in 0..n {
            residual = residual.max((av[(i, j)] - eig.vectors[(i, j)] * eig.values[j]).abs());
        }
    }
    let gram = eig.vectors.transpose().matmul(&eig.vectors)?;
    let orthogonal = gram.max_abs_diff(&Matrix::identity(n));
    assert!(residual < 1e-10 && orthogonal < 1e-11);
    cases.push(json!({"kind":"symmetric_eigen","n":n,"values":eig.values,"residual":residual,
        "orthogonality_error":orthogonal,"timing":latency(||{std::hint::black_box(symmetric_eigen(&spd).unwrap());})}));
    let p = Gfn1Parameters::builtin()?;
    let o = ElectronicOptions {
        electronic_temperature: 0.,
        energy_tolerance: 1e-13,
        charge_tolerance: 1e-11,
        max_scc: 1000,
        ..Default::default()
    };
    let water = PeriodicSystem::from_xyz_str(include_str!("../../examples/water.xyz"), 0., false)?;
    let h = AnalyticHessianOptions {
        electronic_options: o.clone(),
        ..Default::default()
    };
    let direction: Vec<_> = (0..9).map(|i| 0.13 - 0.07 * (i % 4) as f64).collect();
    let cutoff = o.hamiltonian.coordination_cutoff;
    let g = analytic_gradient(
        &water,
        &p,
        AnalyticGradientOptions {
            electronic: o.clone(),
            ..Default::default()
        },
    )?;
    let gradient: Vec<_> = g.gradient.iter().flat_map(|v| v.to_array()).collect();
    let hs = analytic_hessian(&water, &p, h.clone())?.hessian;
    let cubic = directional_third_finite_t(&water, &p, &h, cutoff, &direction)?;
    let quartic = directional_fourth_derivative(&water, &p, &h, cutoff, &direction)?;
    cases.push(json!({"kind":"water_quantum","energy":g.electronic_result.total_free,"gradient":gradient,"hessian":hs.as_slice(),"fc3_directional":cubic,"fc4_directional":quartic,
        "fc3_timing":latency(||{std::hint::black_box(directional_third_finite_t(&water,&p,&h,cutoff,&direction).unwrap());}),
        "fc4_timing":latency(||{std::hint::black_box(directional_fourth_derivative(&water,&p,&h,cutoff,&direction).unwrap());})}));
    let raw = include_str!("../../examples/water48.xyz");
    let xyz = std::iter::once("24")
        .chain(std::iter::once("first eight waters"))
        .chain(raw.lines().skip(2).take(24))
        .collect::<Vec<_>>()
        .join("\n");
    let cluster = PeriodicSystem::from_xyz_str(&xyz, 0., false)?;
    let gc = analytic_gradient(
        &cluster,
        &p,
        AnalyticGradientOptions {
            electronic: o.clone(),
            ..Default::default()
        },
    )?;
    assert!(gc.electronic_result.converged);
    cases.push(json!({"kind":"water24_quantum","energy":gc.electronic_result.total_free,"gradient":gc.gradient.iter().flat_map(|v|v.to_array()).collect::<Vec<_>>(),
        "nao":gc.electronic_result.basis.len(),"scf_iterations":gc.electronic_result.iterations,
        "gradient_timing":latency(||{std::hint::black_box(analytic_gradient(&cluster,&p,AnalyticGradientOptions{electronic:o.clone(),..Default::default()}).unwrap());})}));
    let report = json!({"schema":"gfn1-rs-backend-audit-v1","package_version":env!("CARGO_PKG_VERSION"),
        "lockfile_sha256":format!("{:x}",sha2::Sha256::digest(include_bytes!("../../Cargo.lock"))),
        "os":std::env::consts::OS,"arch":std::env::consts::ARCH,"faer_threads":std::env::var("GFN1_FAER_THREADS").ok(),
        "rayon_threads":rayon::current_num_threads(),"cases":cases});
    std::fs::write(output, serde_json::to_vec_pretty(&report).unwrap())?;
    assert!(
        cases
            .iter()
            .all(|c| c.get("status").is_none_or(|s| s == "pass")),
        "backend audit failure; raw measurements retained"
    );
    println!(
        "backend audit passed: {} numerical/performance cases",
        cases.len()
    );
    Ok(())
}
