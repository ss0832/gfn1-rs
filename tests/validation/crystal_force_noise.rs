// SPDX-License-Identifier: GPL-3.0-or-later
//! Developer-only validation/diagnostic driver; see tests/README.md.
//! Separate finite-difference SCF noise from a fixed analytic Hessian error.
use gfn1_rs::math::Vec3;
use gfn1_rs::*;
fn main() -> Result<()> {
    let data: serde_json::Value = serde_json::from_str(include_str!("../data/real_crystals.json"))
        .map_err(|e| Gfn1Error::InvalidInput(e.to_string()))?;
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    let id = arguments
        .first()
        .cloned()
        .unwrap_or_else(|| "ferrocene_adduct".into());
    let central_z = arguments
        .get(1)
        .map(|v| {
            v.parse::<usize>().map_err(|_| {
                Gfn1Error::InvalidInput("central atomic number must be an integer".into())
            })
        })
        .transpose()?
        .unwrap_or(8);
    let case = data["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["id"] == id)
        .ok_or_else(|| Gfn1Error::InvalidInput("unknown crystal".into()))?;
    let system = PeriodicSystem::from_xyz_str(case["xyz"].as_str().unwrap(), 0., false)?;
    let pair = gfn1_rs::pairlist::unique_short_range_pairs(&system, 3.)?
        .into_iter()
        .filter(|p| system.atoms[p.i].z as usize * system.atoms[p.j].z as usize == central_z)
        .filter(|p| system.atoms[p.i].z == 1 || system.atoms[p.j].z == 1)
        .min_by(|a, b| a.r.total_cmp(&b.r))
        .ok_or_else(|| Gfn1Error::InvalidInput("fixture has no selected XH bond".into()))?;
    let (o, h, sign) = if system.atoms[pair.i].z as usize == central_z {
        (pair.i, pair.j, 1.)
    } else {
        (pair.j, pair.i, -1.)
    };
    let unit = pair.dr * (sign / pair.r);
    let mh = case["masses_amu"][h].as_f64().unwrap();
    let mo = case["masses_amu"][o].as_f64().unwrap();
    let mut direction = vec![Vec3::zero(); system.atoms.len()];
    direction[h] = unit / mh.sqrt();
    direction[o] = unit * (-mh.sqrt() / mo);
    let norm = (mh * direction[h].norm2() + mo * direction[o].norm2()).sqrt();
    for v in &mut direction {
        *v = *v / norm;
    }
    let cutoff = arguments
        .get(2)
        .map(|v| {
            v.parse::<f64>()
                .map_err(|_| Gfn1Error::InvalidInput("invalid AO cutoff".into()))
        })
        .transpose()?
        .unwrap_or(case["ao_cutoff_bohr"].as_f64().unwrap());
    let params = Gfn1Parameters::builtin()?;
    let pbc = PbcOptions {
        ao_cutoff: cutoff,
        ..Default::default()
    };
    println!(
        "crystal {id} Z {central_z} atom {o} H {h} distance_bohr {} cutoff {}",
        pair.r, pbc.ao_cutoff
    );
    for (etol, qtol) in [(1e-11, 1e-9), (1e-13, 1e-12)] {
        let options = ElectronicOptions {
            electronic_temperature: 300.,
            energy_tolerance: etol,
            charge_tolerance: qtol,
            max_scc: 600,
            mixing: 0.25,
            scc_broyden_size: 200,
            ..Default::default()
        };
        for step in [2e-3, 1e-3] {
            let start = std::time::Instant::now();
            let force = |sign: f64| -> Result<f64> {
                let mut changed = system.clone();
                for (atom, v) in changed.atoms.iter_mut().zip(&direction) {
                    atom.position += *v * (sign * step);
                }
                let g = gfn1_rs::pbc::pbc_analytic_gradient(&changed, &params, &options, &pbc)?;
                if !g.scf.converged {
                    return Err(Gfn1Error::InvalidInput("diagnostic SCF failed".into()));
                }
                Ok(g.gradient
                    .iter()
                    .zip(&direction)
                    .map(|(g, v)| g.dot(*v))
                    .sum())
            };
            let fd = (force(1.)? - force(-1.)?) / (2. * step);
            println!("energy_tolerance {etol:e} charge_tolerance {qtol:e} step {step:e} force_wave {fd:e} seconds {}",start.elapsed().as_secs_f64());
        }
    }
    Ok(())
}
