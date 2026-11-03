// SPDX-License-Identifier: GPL-3.0-or-later
//! Developer-only performance profiling driver; see tests/README.md.
//! Reproducible FC3 wall-time / Rust allocator measurement. Set both
//! RAYON_NUM_THREADS=1 and GFN1_FAER_THREADS=1 for a serial comparison.
//! Peak bytes are allocator-tracked live Rust allocations, NOT process RSS.
use gfn1_rs::third_derivative::CubicReference;
use gfn1_rs::*;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};
struct Allocator;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(p, layout) };
    }
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let out = unsafe { System.realloc(p, layout, size) };
        if !out.is_null() {
            let live = if size >= layout.size() {
                LIVE.fetch_add(size - layout.size(), Ordering::Relaxed) + size - layout.size()
            } else {
                LIVE.fetch_sub(layout.size() - size, Ordering::Relaxed) - (layout.size() - size)
            };
            PEAK.fetch_max(live, Ordering::Relaxed);
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        out
    }
}
#[global_allocator]
static GLOBAL: Allocator = Allocator;
fn measure<T>(label: &str, work: impl FnOnce() -> Result<T>) -> Result<T> {
    let start_live = LIVE.load(Ordering::Relaxed);
    PEAK.store(start_live, Ordering::Relaxed);
    ALLOCS.store(0, Ordering::Relaxed);
    let start = Instant::now();
    let result = work()?;
    let seconds = start.elapsed().as_secs_f64();
    println!(
        "{label},{seconds:.9},{},{}",
        PEAK.load(Ordering::Relaxed).saturating_sub(start_live),
        ALLOCS.load(Ordering::Relaxed)
    );
    Ok(result)
}
fn main() -> Result<()> {
    let system = match std::env::args().nth(1) {
        Some(p) => PeriodicSystem::from_xyz_file(p, 0.0, false)?,
        None => PeriodicSystem::from_xyz_str(include_str!("../../examples/water.xyz"), 0.0, false)?,
    };
    let params = Gfn1Parameters::builtin()?;
    let options = ElectronicOptions {
        electronic_temperature: 0.0,
        energy_tolerance: 1e-12,
        charge_tolerance: 1e-10,
        ..Default::default()
    };
    let d = 3 * system.atoms.len();
    let v: Vec<f64> = (0..d).map(|i| 0.15 - 0.07 * (i % 4) as f64).collect();
    // Both routes see a warm shared parameter/D3 cache; SCF counts are explicit.
    drop(run_electronic(&system, &params, options.clone())?);
    println!(
        "# atoms={}, dof={}, rayon_threads={}, faer_threads={:?}",
        system.atoms.len(),
        d,
        rayon::current_num_threads(),
        std::env::var("GFN1_FAER_THREADS").ok()
    );
    println!("stage,wall_seconds,peak_extra_rust_bytes,allocations");
    let legacy = measure("legacy_full_fc3_and_contraction", || {
        let tensor = third_derivative_analytic_dense(
            &system,
            &params,
            AnalyticHessianOptions {
                electronic_options: options.clone(),
                ..Default::default()
            },
            options.hamiltonian.coordination_cutoff,
        )?;
        Ok(tensor.contract_vvv(&v))
    })?;
    let reference = measure("contracted_reference_one_scf_one_lu", || {
        CubicReference::build(&system, &params, &options, 512 * 1024 * 1024)
    })?;
    let direct = measure("contracted_one_direction_one_rhs", || {
        reference.directional(&v)
    })?;
    let second = measure("contracted_reused_reference_second_direction", || {
        reference.directional(&v.iter().map(|x| -x).collect::<Vec<_>>())
    })?;
    println!(
        "# absolute_error={:.9e}, oddness_error={:.9e}, workspace_estimate_bytes={}",
        (direct - legacy).abs(),
        (direct + second).abs(),
        reference.estimated_workspace_bytes
    );
    Ok(())
}
