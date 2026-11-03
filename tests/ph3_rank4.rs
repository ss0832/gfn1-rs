// SPDX-License-Identifier: GPL-3.0-or-later
use gfn1_rs::*;

const PH3: &str =
    "4\nasymmetric phosphine\nP .1 .2 -.1\nH 1.24 .15 .72\nH -.5 1.22 .61\nH -.54 -.91 .66\n";

#[test]
fn high_rank_phosphine_converges_without_changing_temperature() {
    let system = PeriodicSystem::from_xyz_str(PH3, 0., false).unwrap();
    let parameters = Gfn1Parameters::builtin().unwrap();
    let options = ElectronicOptions {
        multipole: true,
        multipole_order: 4,
        electronic_temperature: 0.,
        energy_tolerance: 1e-13,
        charge_tolerance: 1e-11,
        max_scc: 1000,
        ..Default::default()
    };
    let r = run_electronic(&system, &parameters, options).unwrap();
    println!(
        "PH3 rank4: converged={} T={} iterations={} E={:.15}",
        r.converged, r.electronic_temperature, r.iterations, r.total_free
    );
    assert!(r.converged);
    assert_eq!(r.electronic_temperature, 0.);
    assert!(r.occupations.iter().all(|f| *f == 0. || *f == 2.));
    assert!(r.total_free.is_finite());
    let fp = r.fock.matmul(&r.density).unwrap();
    let sw = r
        .integrals
        .overlap
        .matmul(&r.energy_weighted_density)
        .unwrap();
    let residual = fp
        .as_slice()
        .iter()
        .zip(sw.as_slice())
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    assert!(
        residual < 1e-10,
        "returned density/Pulay weight are not stationary: {residual:e}"
    );
    println!(
        "PH3 stationarity={residual:e}; orbital energies={:?}; occupations={:?}",
        r.orbital_energies, r.occupations
    );
    assert!(
        r.total_free < 0.,
        "canonicalization discarded the orbital minimum"
    );
}
