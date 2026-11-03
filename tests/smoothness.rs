// SPDX-License-Identifier: GPL-3.0-or-later
use gfn1_rs::math::Vec3;
use gfn1_rs::smoothness::NuclearNeighborhood;
use gfn1_rs::*;
#[test]
fn periodic_cutoff_domain_distinguishes_nuclear_motion_from_branch_crossing() {
    let system = PeriodicSystem::from_xyz_str(
        "2\nLattice=\"17.37 0 0 .2 19.43 0 .1 .3 20.89\" pbc=\"T T T\"\nH 0 0 0\nH 4.00001 0 0\n",
        0.,
        true,
    )
    .unwrap();
    let options = ElectronicOptions::default();
    let pbc = PbcOptions {
        ao_cutoff: 4.,
        ..Default::default()
    };
    let domain = NuclearNeighborhood::periodic_scalar(&system, &options, &pbc).unwrap();
    let mut translated = system.clone();
    for atom in &mut translated.atoms {
        atom.position += Vec3::new(0.123, 0.276, -0.311);
    }
    assert!(domain.changed_terms(&translated).unwrap().is_empty());
    let mut displaced = system.clone();
    displaced.atoms[1].position.x -= 1e-3;
    assert!(domain.changed_terms(&displaced).unwrap().contains(&"ao"));
    displaced.atoms[0].z = 2;
    assert!(domain.changed_terms(&displaced).is_err());
}
