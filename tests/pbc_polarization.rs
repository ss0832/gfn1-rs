// SPDX-License-Identifier: GPL-3.0-or-later
//! Gates for the Berry-phase bulk polarization
//! (`gfn1_rs::pbc::polarization::pbc_berry_polarization`).
//!
//! The properties that pin the implementation down:
//!
//! 1. **Molecular limit.** A polar molecule in a large box, Γ-only Resta: `P * V`
//!    must approach the exact quantum-mechanical dipole `Σ_A z_A R_A − Tr[P r]`
//!    (the `r` integrals come from `lao_dipole_matrix` at `B = 0`) as the box
//!    grows, and it must do so like `1/L²`.
//! 2. **String refinement.** At fixed box, refining the KSV string converges onto
//!    the same exact dipole like `1/N²` — the gate on the link bookkeeping itself.
//! 3. **Quantum consistency.** Translating *every* atom by one lattice vector
//!    leaves the electronic Berry phase untouched and shifts the raw total phase by
//!    exactly `2 π N_el` — whole polarization quanta — so the reduced `P` is
//!    unchanged.
//! 4. **Inversion symmetry.** A centrosymmetric crystal (diamond, rock-salt NaCl)
//!    has `P = 0` modulo *half* the spin-restricted quantum, i.e. `Φ ≡ 0 (mod 2 π)`.
//! 5. **Negative control + Born charge.** Diamond's Born charges vanish by
//!    symmetry, so the "the phase actually moves" control uses heteropolar NaCl: a
//!    polar sublattice shift must respond, linearly, with a physical `Z*`.
//! 6. **KSV ↔ Resta.** A one-point KSV string must reproduce the Resta value.
//! 7. **Fractional occupations (and charged cells) are rejected**, with the error
//!    texts quoted in `docs/limitations.md`.

use gfn1_rs::pbc::polarization::{
    pbc_berry_polarization, BerryMethodSelector, BerryPolarizationMethod, BerryPolarizationOptions,
    POLARIZATION_AU_TO_C_PER_M2,
};
use gfn1_rs::{
    lao_dipole_matrix, run_electronic_pbc, ElectronicOptions, ExternalFieldOptions, Gfn1Parameters,
    PbcOptions, PeriodicSystem,
};
use std::f64::consts::PI;

fn params() -> Gfn1Parameters {
    Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed")
}

/// Hydrogen fluoride centred in a cubic box of edge `l` Angstrom, aligned with z.
fn hf_box(l: f64) -> PeriodicSystem {
    let c = 0.5 * l;
    let half = 0.5 * 0.917;
    let text = format!(
        "2\nLattice=\"{l} 0 0 0 {l} 0 0 0 {l}\" pbc=\"T T T\"\n\
         F {c:.6} {c:.6} {:.6}\n\
         H {c:.6} {c:.6} {:.6}\n",
        c - half,
        c + half
    );
    PeriodicSystem::from_xyz_str(&text, 0.0, false).unwrap()
}

/// Rock-salt NaCl, conventional 8-atom cubic cell (`a = 5.64 A`). Centrosymmetric,
/// but *heteropolar*, so unlike diamond it has non-vanishing Born charges — which
/// makes it the right fixture for the "does the phase actually move" control.
const NACL: &str = "8\n\
Lattice=\"5.64 0 0 0 5.64 0 0 0 5.64\" pbc=\"T T T\"\n\
Na 0.00 0.00 0.00\n\
Na 0.00 2.82 2.82\n\
Na 2.82 0.00 2.82\n\
Na 2.82 2.82 0.00\n\
Cl 2.82 2.82 2.82\n\
Cl 2.82 0.00 0.00\n\
Cl 0.00 2.82 0.00\n\
Cl 0.00 0.00 2.82\n";

/// NaCl with the whole Na sublattice pushed by `shift` Angstrom along z.
fn nacl_polar(shift: f64) -> PeriodicSystem {
    let mut system = PeriodicSystem::from_xyz_str(NACL, 0.0, false).unwrap();
    let dz = shift * 1.889_726_124_625_770_2;
    for atom in system.atoms.iter_mut() {
        if atom.z == 11 {
            atom.position.z += dz;
        }
    }
    system
}

const DIAMOND: &str = "8\n\
Lattice=\"3.567 0 0 0 3.567 0 0 0 3.567\" pbc=\"T T T\"\n\
C 0.000000 0.000000 0.000000\n\
C 0.891750 0.891750 0.891750\n\
C 0.000000 1.783500 1.783500\n\
C 0.891750 2.675250 2.675250\n\
C 1.783500 0.000000 1.783500\n\
C 2.675250 0.891750 2.675250\n\
C 1.783500 1.783500 0.000000\n\
C 2.675250 2.675250 0.891750\n";

/// Exact dipole `Σ_A z_A R_A − Tr[P r]` (e·bohr) of the periodic Γ density, using
/// the same AO dipole integrals `lao_dipole_matrix` provides at zero field. This is
/// the *quantum-mechanical* dipole (not the Mulliken point-charge one), which is
/// what a Berry phase reproduces in the isolated-molecule limit.
fn exact_gamma_dipole(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
) -> [f64; 3] {
    let res = run_electronic_pbc(system, params, options).unwrap();
    assert!(res.converged, "reference SCC did not converge");
    let n = res.basis.len();
    let d = lao_dipole_matrix(system, &res.basis, &ExternalFieldOptions::default());
    let mut dipole = [0.0_f64; 3];
    for (c, dc) in d.iter().enumerate() {
        let mut tr = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                tr += res.density[(i, j)] * dc.re[(j, i)];
            }
        }
        dipole[c] = -tr;
    }
    for (a, atom) in system.atoms.iter().enumerate() {
        let z = res.basis.reference_electrons[a];
        let r = atom.position.to_array();
        for c in 0..3 {
            dipole[c] += z * r[c];
        }
    }
    dipole
}

fn zero_temperature() -> ElectronicOptions {
    ElectronicOptions {
        electronic_temperature: 0.0,
        ..ElectronicOptions::default()
    }
}

// ---------------------------------------------------------------------------
// 1. Molecular limit
// ---------------------------------------------------------------------------

#[test]
fn resta_dipole_approaches_the_exact_molecular_dipole_as_the_box_grows() {
    let params = params();
    let options = zero_temperature();
    let pbc = PbcOptions::default();
    let berry = BerryPolarizationOptions::default();

    let mut row = Vec::new();
    for &l in &[8.0_f64, 12.0_f64, 16.0_f64] {
        let system = hf_box(l);
        let reference = exact_gamma_dipole(&system, &params, &options);
        let result = pbc_berry_polarization(&system, &params, &options, &pbc, &berry).unwrap();
        assert_eq!(result.method, BerryPolarizationMethod::Resta);
        let err: f64 = (0..3)
            .map(|c| (result.dipole[c] - reference[c]).abs())
            .fold(0.0, f64::max);
        println!(
            "L = {l:>5.1} A  V = {:>12.3} a0^3  Berry mu_z = {:+.8}  exact mu_z = {:+.8}  \
             max|d| = {err:.4e}",
            result.volume, result.dipole[2], reference[2]
        );
        row.push((l, err, result.dipole[2], reference[2]));
    }
    for w in row.windows(2) {
        let scale = w[1].0 / w[0].0;
        println!(
            "  L {:.0} -> {:.0} A (x{scale:.3}): residual {:.4e} -> {:.4e}, ratio {:.3} \
             (1/L^2 predicts {:.3})",
            w[0].0,
            w[1].0,
            w[0].1,
            w[1].1,
            w[0].1 / w[1].1,
            scale * scale
        );
    }

    // The dipole must be recovered, and the residual must shrink like 1/L^2: the
    // Resta single-point form is exact only as `b = 2 pi / L -> 0`, its leading
    // error being O(b^2) times the second moment of the occupied orbitals.
    for w in row.windows(2) {
        let observed = w[0].1 / w[1].1;
        let predicted = (w[1].0 / w[0].0).powi(2);
        assert!(
            observed > 0.75 * predicted && observed < 1.35 * predicted,
            "molecular-limit residual ratio {observed:.3} is not the 1/L^2 trend {predicted:.3} \
             ({:.3e} at L = {} -> {:.3e} at L = {})",
            w[0].1,
            w[0].0,
            w[1].1,
            w[1].0
        );
    }
    assert!(
        row[2].1 < 4.0e-3,
        "largest-box Berry dipole off the exact dipole by {:.3e} e a0",
        row[2].1
    );
    // Every box must at least get the polar axis right to a few percent.
    for (l, _, berry_z, exact_z) in &row {
        assert!(
            (berry_z - exact_z).abs() < 0.02 * exact_z.abs(),
            "L = {l}: Berry mu_z {berry_z:+.6} vs exact {exact_z:+.6}"
        );
    }
}

#[test]
fn refining_the_ksv_string_converges_onto_the_exact_molecular_dipole() {
    let params = params();
    let options = zero_temperature();
    let pbc = PbcOptions::default();
    let system = hf_box(8.0);
    let reference = exact_gamma_dipole(&system, &params, &options)[2];

    // Along z the string carries `n` points, so the boost per link is `b_z / n`
    // and the discretisation error should fall like 1/n^2. Everything else about
    // the system is unchanged, so this isolates the string machinery.
    let mut errors = Vec::new();
    for &n in &[1_usize, 2, 4] {
        let berry = BerryPolarizationOptions {
            mesh: [1, 1, n],
            method: BerryMethodSelector::KingSmithVanderbilt,
            directions: [false, false, true],
            ..BerryPolarizationOptions::default()
        };
        let result = pbc_berry_polarization(&system, &params, &options, &pbc, &berry).unwrap();
        let err = (result.dipole[2] - reference).abs();
        println!(
            "string points = {n}: mu_z = {:+.8} (exact {reference:+.8}), residual {err:.4e}",
            result.dipole[2]
        );
        errors.push(err);
    }
    assert!(
        errors[1] < errors[0] && errors[2] < errors[1],
        "KSV string refinement did not converge: {errors:?}"
    );
    assert!(
        errors[2] < 0.2 * errors[0],
        "4-point string residual {:.3e} is not a clear refinement of the 1-point (Resta) \
         residual {:.3e}",
        errors[2],
        errors[0]
    );
}

// ---------------------------------------------------------------------------
// 2. Quantum consistency under a lattice translation
// ---------------------------------------------------------------------------

#[test]
fn lattice_translation_shifts_the_raw_phase_by_whole_quanta() {
    let params = params();
    let options = zero_temperature();
    let pbc = PbcOptions::default();
    let berry = BerryPolarizationOptions::default();

    let l = 8.0_f64;
    let base = hf_box(l);
    let reference = pbc_berry_polarization(&base, &params, &options, &pbc, &berry).unwrap();

    // Translate EVERY atom by the first lattice vector.
    let mut shifted = base.clone();
    let a1 = base.lattice.as_ref().unwrap().cell.col[0];
    for atom in shifted.atoms.iter_mut() {
        atom.position += a1;
    }
    let moved = pbc_berry_polarization(&shifted, &params, &options, &pbc, &berry).unwrap();

    let nelec: f64 = reference.occupied_bands as f64 * 2.0;
    let delta = moved.total_phase_raw[0] - reference.total_phase_raw[0];
    let quanta = delta / (2.0 * PI);
    println!(
        "translation: raw phase {:.9} -> {:.9} (delta {:.9} = {:.6} x 2 pi, N_el = {nelec})",
        reference.total_phase_raw[0], moved.total_phase_raw[0], delta, quanta
    );
    println!(
        "  electronic phase {:.12} -> {:.12}",
        reference.electronic_phase[0], moved.electronic_phase[0]
    );
    assert!(
        (quanta - nelec).abs() < 1.0e-9,
        "raw phase shifted by {quanta} quanta, expected exactly N_el = {nelec}"
    );
    // The electronic Berry phase itself is translation invariant.
    for c in 0..3 {
        assert!(
            (moved.electronic_phase[c] - reference.electronic_phase[c]).abs() < 1.0e-8,
            "electronic phase moved on axis {c}: {:.12} -> {:.12}",
            reference.electronic_phase[c],
            moved.electronic_phase[c]
        );
    }
    // Hence P (mod quantum) and the reduced phase are unchanged.
    for c in 0..3 {
        assert!(
            (moved.total_phase_reduced[c] - reference.total_phase_reduced[c]).abs() < 1.0e-7,
            "reduced phase moved on axis {c}"
        );
        assert!(
            (moved.polarization[c] - reference.polarization[c]).abs() < 1.0e-9,
            "polarization moved on axis {c}"
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Inversion symmetry
// ---------------------------------------------------------------------------

#[test]
fn centrosymmetric_diamond_polarization_vanishes_modulo_half_the_quantum() {
    let params = params();
    let options = zero_temperature();
    let pbc = PbcOptions::default();
    let system = PeriodicSystem::from_xyz_str(DIAMOND, 0.0, false).unwrap();

    for (label, berry) in [
        ("Resta", BerryPolarizationOptions::default()),
        (
            "KSV[2,2,2]",
            BerryPolarizationOptions {
                mesh: [2, 2, 2],
                method: BerryMethodSelector::KingSmithVanderbilt,
                ..BerryPolarizationOptions::default()
            },
        ),
    ] {
        let result = pbc_berry_polarization(&system, &params, &options, &pbc, &berry).unwrap();
        for d in 0..3 {
            // Half the spin-restricted quantum <=> Phi = 0 (mod 2 pi).
            let residual = {
                let x = result.total_phase_raw[d];
                let r = x - 2.0 * PI * (x / (2.0 * PI) + 0.5).floor();
                if r <= -PI {
                    r + 2.0 * PI
                } else {
                    r
                }
            };
            println!(
                "{label} d={d}: phi_el = {:+.9}  phi_ion = {:+.9}  Phi_raw = {:+.9}  \
                 residual (mod 2 pi) = {residual:+.3e}",
                result.electronic_phase[d], result.ionic_phase[d], result.total_phase_raw[d]
            );
            assert!(
                residual.abs() < 1.0e-6,
                "{label}: diamond is centrosymmetric but Phi_{d} = {:+.9} leaves a residual \
                 {residual:+.3e} modulo 2 pi",
                result.total_phase_raw[d]
            );
        }
        println!(
            "{label}: P = [{:+.3e}, {:+.3e}, {:+.3e}] e/a0^2, quantum along a1 = \
             [{:+.4e}, {:+.4e}, {:+.4e}]",
            result.polarization[0],
            result.polarization[1],
            result.polarization[2],
            result.quantum[0][0],
            result.quantum[0][1],
            result.quantum[0][2]
        );
    }
}

#[test]
fn centrosymmetric_rocksalt_vanishes_but_a_polar_distortion_does_not() {
    let params = params();
    let options = zero_temperature();
    let pbc = PbcOptions::default();
    // Only the z direction is needed here, which keeps the boosted-overlap image
    // sums to one build per structure.
    let berry = BerryPolarizationOptions {
        mesh: [2, 2, 2],
        method: BerryMethodSelector::KingSmithVanderbilt,
        directions: [false, false, true],
        ..BerryPolarizationOptions::default()
    };

    let mut phases = Vec::new();
    for &shift in &[0.0_f64, 0.02, 0.04] {
        let system = nacl_polar(shift);
        let result = pbc_berry_polarization(&system, &params, &options, &pbc, &berry).unwrap();
        println!(
            "NaCl, Na sublattice +{shift:.3} A along z: phi_el = {:+.9}  Phi_reduced = {:+.9}  \
             P_z = {:+.6e} e/a0^2  ({:+.4e} C/m^2)  quantum_z = {:+.4e}",
            result.electronic_phase[2],
            result.total_phase_reduced[2],
            result.polarization[2],
            result.polarization[2] * POLARIZATION_AU_TO_C_PER_M2,
            result.quantum[2][2]
        );
        phases.push((shift, result.total_phase_reduced[2], result.polarization[2]));
    }

    // Undistorted rock salt is centrosymmetric: P = 0 modulo half the quantum.
    assert!(
        phases[0].1.abs() < 1.0e-6,
        "centrosymmetric NaCl left a reduced phase {:+.3e}",
        phases[0].1
    );
    // A polar sublattice shift must move the phase (the negative control: the gate
    // above is not passing because the machinery returns zero for everything), and
    // must do so linearly — that slope is the Born effective charge of Na.
    assert!(
        phases[1].1.abs() > 1.0e-4,
        "a 0.02 A polar distortion left the phase at {:+.3e}; the Berry phase is not \
         responding to the structure",
        phases[1].1
    );
    let ratio = phases[2].1 / phases[1].1;
    println!("  linearity: Phi(0.04)/Phi(0.02) = {ratio:.4} (harmonic limit 2)");
    assert!(
        (ratio - 2.0).abs() < 0.1,
        "polar-distortion response is not linear: ratio {ratio:.4}"
    );

    // The slope is the Born effective charge of Na: dP_z * V = Z* * n_Na * du_z.
    let volume = pbc_berry_polarization(&nacl_polar(0.0), &params, &options, &pbc, &berry)
        .unwrap()
        .volume;
    let du = 0.04 * 1.889_726_124_625_770_2;
    let z_born = (phases[2].2 - phases[0].2) * volume / (4.0 * du);
    println!("  Born effective charge Z*(Na) = {z_born:+.4} e (nominal ionic +1)");
    assert!(
        z_born > 0.5 && z_born < 1.5,
        "Z*(Na) = {z_born:+.4} is not a physical Born charge for rock-salt NaCl"
    );
}

// ---------------------------------------------------------------------------
// 4. KSV with one k-point per string == Resta
// ---------------------------------------------------------------------------

#[test]
fn one_point_ksv_string_reproduces_the_resta_single_point_form() {
    let params = params();
    let options = zero_temperature();
    let pbc = PbcOptions::default();
    let system = hf_box(8.0);

    let resta = pbc_berry_polarization(
        &system,
        &params,
        &options,
        &pbc,
        &BerryPolarizationOptions {
            method: BerryMethodSelector::Resta,
            ..BerryPolarizationOptions::default()
        },
    )
    .unwrap();
    let ksv = pbc_berry_polarization(
        &system,
        &params,
        &options,
        &pbc,
        &BerryPolarizationOptions {
            mesh: [1, 1, 1],
            method: BerryMethodSelector::KingSmithVanderbilt,
            ..BerryPolarizationOptions::default()
        },
    )
    .unwrap();
    assert_eq!(resta.method, BerryPolarizationMethod::Resta);
    assert_eq!(ksv.method, BerryPolarizationMethod::KingSmithVanderbilt);
    for d in 0..3 {
        let dphi = (ksv.electronic_phase[d] - resta.electronic_phase[d]).abs();
        println!(
            "d={d}: Resta phi = {:+.14}  KSV(N=1) phi = {:+.14}  |diff| = {dphi:.3e}",
            resta.electronic_phase[d], ksv.electronic_phase[d]
        );
        assert!(
            dphi < 1.0e-10,
            "one-point KSV string differs from Resta on axis {d} by {dphi:.3e}"
        );
    }
    for c in 0..3 {
        assert!((ksv.dipole[c] - resta.dipole[c]).abs() < 1.0e-9);
    }
}

// ---------------------------------------------------------------------------
// 5. Fractional occupations are rejected
// ---------------------------------------------------------------------------

#[test]
fn fractional_occupations_are_rejected() {
    let params = params();
    let system = hf_box(8.0);
    let options = ElectronicOptions {
        electronic_temperature: 50_000.0,
        ..ElectronicOptions::default()
    };
    let err = pbc_berry_polarization(
        &system,
        &params,
        &options,
        &PbcOptions::default(),
        &BerryPolarizationOptions::default(),
    )
    .unwrap_err();
    let text = err.to_string();
    println!("rejection: {text}");
    assert!(
        text.contains("requires integer band occupations"),
        "unexpected error text: {text}"
    );
}

#[test]
fn open_shell_and_charged_cells_are_rejected() {
    let params = params();
    let options = zero_temperature();
    let charged = PeriodicSystem::from_xyz_str(
        "2\nLattice=\"8 0 0 0 8 0 0 0 8\" pbc=\"T T T\"\nF 4.0 4.0 3.54\nH 4.0 4.0 4.46\n",
        -1.0,
        false,
    )
    .unwrap();
    let err = pbc_berry_polarization(
        &charged,
        &params,
        &options,
        &PbcOptions::default(),
        &BerryPolarizationOptions::default(),
    )
    .unwrap_err()
    .to_string();
    println!("charged rejection: {err}");
    assert!(err.contains("requires a neutral cell"), "{err}");
}
