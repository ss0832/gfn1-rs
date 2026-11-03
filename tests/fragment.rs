// SPDX-License-Identifier: GPL-3.0-or-later
use gfn1_rs::fragment::{
    divide_and_conquer, fmo_xtb, CovalentCut, DivideConquerOptions, FmoOptions, FragmentPartition,
};
use gfn1_rs::math::{Mat3, Vec3};
use gfn1_rs::{ElectronicOptions, Gfn1Parameters, PbcOptions, PeriodicSystem};

fn options(t: f64) -> ElectronicOptions {
    ElectronicOptions {
        electronic_temperature: t,
        max_scc: 1000,
        energy_tolerance: 1e-13,
        charge_tolerance: 1e-11,
        ..Default::default()
    }
}
fn move_coordinate(system: &PeriodicSystem, y: usize, h: f64) -> PeriodicSystem {
    let mut out = system.clone();
    let ncart = 3 * system.atoms.len();
    if y < ncart {
        let mut r = out.atoms[y / 3].position.to_array();
        r[y % 3] += h;
        out.atoms[y / 3].position = Vec3::new(r[0], r[1], r[2]);
    } else {
        let a = (y - ncart) / 3;
        let b = (y - ncart) % 3;
        let strain = |r: Vec3| {
            let mut v = r.to_array();
            v[a] += h * v[b];
            Vec3::new(v[0], v[1], v[2])
        };
        for atom in &mut out.atoms {
            atom.position = strain(atom.position);
        }
        let lattice = out.lattice.unwrap();
        out.lattice = Some(
            gfn1_rs::lattice::Lattice::new(
                Mat3 {
                    col: lattice.cell.col.map(strain),
                },
                lattice.periodic,
            )
            .unwrap(),
        );
    }
    out
}
fn trimer(periodic: bool) -> PeriodicSystem {
    let comment = if periodic {
        "Lattice=\"15 .3 .2 0 16 .3 .1 0 17\" pbc=\"T T T\""
    } else {
        "HF trimer"
    };
    PeriodicSystem::from_xyz_str(&format!("6\n{comment}\nH .1 .2 .3\nF 1.9 .3 .4\nH .4 5.3 .2\nF 1.8 5.8 .6\nH 5.4 1.1 .7\nF 6.8 1.7 .8\n"),0.,true).unwrap()
}
fn partition() -> FragmentPartition {
    FragmentPartition {
        atoms: vec![vec![0, 1], vec![2, 3], vec![4, 5]],
        charges: vec![0.; 3],
        cuts: vec![],
    }
}

#[test]
fn fragment_complete_limits_recover_gfn1_energy_and_gradient() {
    let params = Gfn1Parameters::builtin().unwrap();
    let pbc = PbcOptions::default();
    for t in [0., 30000.] {
        let system = trimer(false);
        let electronic = options(t);
        let reference = gfn1_rs::analytic_gradient(
            &system,
            &params,
            gfn1_rs::AnalyticGradientOptions {
                electronic: electronic.clone(),
                ..Default::default()
            },
        )
        .unwrap();
        let fmo = fmo_xtb(
            &system,
            &params,
            &electronic,
            &pbc,
            &partition(),
            &FmoOptions {
                order: 3,
                ..Default::default()
            },
        )
        .unwrap();
        let dc = divide_and_conquer(
            &system,
            &params,
            &electronic,
            &pbc,
            &partition(),
            &DivideConquerOptions {
                buffers: vec![(0..6).collect(); 3],
                ..Default::default()
            },
        )
        .unwrap();
        for (name, result) in [("FMO3", fmo), ("DC", dc)] {
            let error = (result.total_free - reference.electronic_result.total_free).abs();
            let gradient = result
                .gradient
                .iter()
                .zip(&reference.gradient)
                .map(|(a, b)| (*a - *b).norm())
                .fold(0.0_f64, f64::max);
            eprintln!("{name} complete t={t} energy={error:.3e} gradient={gradient:.3e}");
            assert!(error < 1e-9 && gradient < 2e-8);
        }
    }
}

#[test]
fn fragment_fmo_and_dc_all_gradients_and_periodic_stress_match_fd() {
    let params = Gfn1Parameters::builtin().unwrap();
    let pbc = PbcOptions::default();
    let electronic = options(30000.);
    for periodic in [false, true] {
        let system = trimer(periodic);
        let fmo_options = FmoOptions {
            embedding_tolerance: 1e-11,
            ..Default::default()
        };
        let dc_options = DivideConquerOptions {
            buffers: vec![vec![0, 1, 2, 3], vec![0, 1, 2, 3, 4, 5], vec![2, 3, 4, 5]],
            ..Default::default()
        };
        for dc in [false, true] {
            let base = if dc {
                divide_and_conquer(
                    &system,
                    &params,
                    &electronic,
                    &pbc,
                    &partition(),
                    &dc_options,
                )
            } else {
                fmo_xtb(
                    &system,
                    &params,
                    &electronic,
                    &pbc,
                    &partition(),
                    &fmo_options,
                )
            }
            .unwrap();
            let ncart = system.atoms.len() * 3;
            let ndof = ncart + if periodic { 9 } else { 0 };
            let mut error = 0.0_f64;
            for y in 0..ndof {
                let analytic = if y < ncart {
                    base.gradient[y / 3].to_array()[y % 3]
                } else {
                    base.stress.as_ref().unwrap()[((y - ncart) / 3, (y - ncart) % 3)]
                        * system.lattice.unwrap().volume()
                };
                // The derivative holds the neighbor topology fixed. A smaller
                // strain is needed here because a 2a image lies just outside
                // the stock 30-bohr CN cutoff in this skew reference cell.
                let h = if y < ncart { 3e-4 } else { 2e-5 };
                let energy = |h: f64| {
                    let displaced = move_coordinate(&system, y, h);
                    if dc {
                        divide_and_conquer(
                            &displaced,
                            &params,
                            &electronic,
                            &pbc,
                            &partition(),
                            &DivideConquerOptions {
                                compute_derivatives: false,
                                ..dc_options.clone()
                            },
                        )
                        .unwrap()
                        .total_free
                    } else {
                        fmo_xtb(
                            &displaced,
                            &params,
                            &electronic,
                            &pbc,
                            &partition(),
                            &FmoOptions {
                                compute_derivatives: false,
                                ..fmo_options.clone()
                            },
                        )
                        .unwrap()
                        .total_free
                    }
                };
                let coarse = (energy(h) - energy(-h)) / (2. * h);
                let fine = (energy(h / 2.) - energy(-h / 2.)) / h;
                if (analytic - (4. * fine - coarse) / 3.).abs() > 1e-8 {
                    eprintln!("DC={dc} periodic={periodic} y={y} analytic={analytic:.12e} FD(h)={coarse:.12e} FD(h/2)={fine:.12e}");
                }
                error = error.max((analytic - (4. * fine - coarse) / 3.).abs());
            }
            eprintln!("fragment DC={dc} periodic={periodic} all {ndof} derivative components max FD error={error:.3e}");
            assert!(error < 2e-7, "DC={dc} periodic={periodic} error={error}");
        }
    }
}

#[test]
fn fragment_covalent_hop_propane_converges_and_matches_displaced_energy() {
    let params = Gfn1Parameters::builtin().unwrap();
    let pbc = PbcOptions::default();
    let electronic = options(0.);
    let system=PeriodicSystem::from_xyz_str("11\npropane\nC -1.5 0 0\nC 0 .3 0\nC 1.4 -.3 .2\nH -1.8 1 .2\nH -1.8 -.4 .9\nH -1.8 -.6 -.8\nH .1 1.3 .2\nH 0 .1 -1.1\nH 1.6 -1.2 -.3\nH 2.2 .4 .1\nH 1.4 -.5 1.3\n",0.,false).unwrap();
    let partition = FragmentPartition {
        atoms: vec![vec![0, 3, 4, 5], vec![1, 6, 7], vec![2, 8, 9, 10]],
        charges: vec![0.; 3],
        cuts: vec![
            CovalentCut {
                bda: 0,
                baa: 1,
                image: [0; 3],
            },
            CovalentCut {
                bda: 2,
                baa: 1,
                image: [0; 3],
            },
        ],
    };
    let settings = FmoOptions::default();
    let result = fmo_xtb(&system, &params, &electronic, &pbc, &partition, &settings).unwrap();
    let h = 1e-3;
    let mut max = 0.0_f64;
    for y in 0..system.atoms.len() * 3 {
        let energy = |h: f64| {
            fmo_xtb(
                &move_coordinate(&system, y, h),
                &params,
                &electronic,
                &pbc,
                &partition,
                &FmoOptions {
                    compute_derivatives: false,
                    ..settings.clone()
                },
            )
            .unwrap()
            .total_free
        };
        let fd = (energy(h) - energy(-h)) / (2. * h);
        max = max.max((result.gradient[y / 3].to_array()[y % 3] - fd).abs());
    }
    eprintln!("propane HOP all 33 gradient components max FD error={max:.3e}");
    assert!(max < 1e-5);
}

#[test]
fn fragment_periodic_covalent_hop_all_forces_stress_and_image_wrapping() {
    let params = Gfn1Parameters::builtin().unwrap();
    let pbc = PbcOptions::default();
    let electronic = options(0.);
    let mut system=PeriodicSystem::from_xyz_str("11\npropane periodic\nC -1.5 0 0\nC 0 .3 0\nC 1.4 -.3 .2\nH -1.8 1 .2\nH -1.8 -.4 .9\nH -1.8 -.6 -.8\nH .1 1.3 .2\nH 0 .1 -1.1\nH 1.6 -1.2 -.3\nH 2.2 .4 .1\nH 1.4 -.5 1.3\n",0.,false).unwrap();
    system.lattice = Some(
        gfn1_rs::lattice::Lattice::new(
            Mat3 {
                col: [
                    Vec3::new(35., 0.4, 0.2),
                    Vec3::new(0.1, 36., 0.3),
                    Vec3::new(0.2, 0.1, 37.),
                ],
            },
            [true; 3],
        )
        .unwrap(),
    );
    let mut partition = FragmentPartition {
        atoms: vec![vec![0, 3, 4, 5], vec![1, 6, 7], vec![2, 8, 9, 10]],
        charges: vec![0.; 3],
        cuts: vec![
            CovalentCut {
                bda: 0,
                baa: 1,
                image: [0; 3],
            },
            CovalentCut {
                bda: 2,
                baa: 1,
                image: [0; 3],
            },
        ],
    };
    let settings = FmoOptions::default();
    let reference = fmo_xtb(&system, &params, &electronic, &pbc, &partition, &settings).unwrap();
    let translation = system.lattice.unwrap().cell.col[0];
    for &a in &partition.atoms[0] {
        system.atoms[a].position += translation;
    }
    partition.cuts[0].image = [1, 0, 0];
    let result = fmo_xtb(&system, &params, &electronic, &pbc, &partition, &settings).unwrap();
    assert!((result.total_free - reference.total_free).abs() < 1e-9);
    let mut maximum = [0.0_f64; 2];
    let ncart = system.atoms.len() * 3;
    for y in 0..ncart + 9 {
        let h = if y < ncart { 5e-4 } else { 2e-5 };
        let energy = |h: f64| {
            fmo_xtb(
                &move_coordinate(&system, y, h),
                &params,
                &electronic,
                &pbc,
                &partition,
                &FmoOptions {
                    compute_derivatives: false,
                    ..settings.clone()
                },
            )
            .unwrap()
            .total_free
        };
        let coarse = (energy(h) - energy(-h)) / (2. * h);
        let fine = (energy(h / 2.) - energy(-h / 2.)) / h;
        let (kind, analytic) = if y < ncart {
            (0, result.gradient[y / 3].to_array()[y % 3])
        } else {
            (
                1,
                result.stress.as_ref().unwrap()[((y - ncart) / 3, (y - ncart) % 3)]
                    * system.lattice.unwrap().volume(),
            )
        };
        maximum[kind] = maximum[kind].max((analytic - (4. * fine - coarse) / 3.).abs());
    }
    eprintln!("periodic covalent HOP all 33 forces/9 strain derivatives max errors {maximum:?}");
    assert!(maximum[0] < 1e-6 && maximum[1] < 3e-6);
}
