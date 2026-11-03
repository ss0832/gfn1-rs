// SPDX-License-Identifier: GPL-3.0-or-later
//! Developer-only validation/diagnostic driver; see tests/README.md.
//! Diagnose AO-sum convergence independently of SCC or force-constant fitting.
use gfn1_rs::pbc::{bloch::BlochBuilder, complex::native_hermitian_eigen};
use gfn1_rs::*;
fn main() -> Result<()> {
    let data: serde_json::Value = serde_json::from_str(include_str!("../data/real_crystals.json"))
        .map_err(|e| Gfn1Error::InvalidInput(e.to_string()))?;
    let id = std::env::args().nth(1).unwrap_or_else(|| "srtio3".into());
    let case = data["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["id"] == id)
        .ok_or_else(|| Gfn1Error::InvalidInput("unknown crystal fixture".into()))?;
    let system = PeriodicSystem::from_xyz_str(case["xyz"].as_str().unwrap(), 0., false)?;
    let params = Gfn1Parameters::builtin()?;
    let basis = BasisSet::build(&system, &params, Default::default())?;
    let minimum = gfn1_rs::basis::atom_min_exponents(&basis, system.atoms.len());
    println!(
        "crystal {id} basis {} minimum_exponents {minimum:?}",
        basis.len()
    );
    for cutoff in [20., 24., 30., 40., 50., 60., 80., 90.] {
        let start = std::time::Instant::now();
        let bloch = BlochBuilder::build(&system, &basis, &params, cutoff, 40., false)?;
        for q in [[0., 0., 0.], [0.5, 0., 0.]] {
            let (_, overlap) = bloch.h_s_at_k(q);
            let eigen = native_hermitian_eigen(&overlap)?;
            println!(
                "cutoff {cutoff} k {q:?} min_overlap {:e} max_overlap {:e} seconds {}",
                eigen.values[0],
                eigen.values[eigen.values.len() - 1],
                start.elapsed().as_secs_f64()
            );
        }
    }
    Ok(())
}
