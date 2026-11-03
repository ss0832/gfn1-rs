// SPDX-License-Identifier: GPL-3.0-or-later

use crate::fragment::{self, CovalentCut, DivideConquerOptions, FmoOptions, FragmentPartition};
use crate::magnetic;
use crate::math::Vec3;
use crate::{
    BerryPolarizationOptions, ElectronicOptions, Gfn1Error, Gfn1Parameters, PbcOptions,
    PeriodicSystem, Result, SecondaryBasis, TdaOptions,
};
use serde::Deserialize;
use serde_json::json;

pub(super) enum Observable {
    Magnetizability,
    MagneticPolarizability,
    CottonMouton,
    StaticMcd,
    Ecd,
    OpticalRotation(f64),
    Piezoelectric,
    Dielectric,
    BerryPolarization,
    Electromechanical,
    Gruneisen(bool),
    StrainHessian(f64),
    Vibrations,
    Fragments { path: String, divide: bool },
}

pub(super) struct Settings<'a> {
    pub pbc: PbcOptions,
    pub tda: TdaOptions,
    pub bfield: Option<Vec3>,
    pub e_step: f64,
    pub b_step: Option<f64>,
    pub m1_basis: Option<&'a str>,
    pub print_matrices: bool,
}

pub(super) fn load_m1_basis(path: Option<&str>) -> Result<Option<SecondaryBasis>> {
    path.map(|name| {
        if let Some(basis) = crate::builtin_secondary(name) {
            basis
        } else {
            crate::parse_secondary_basis(&std::fs::read_to_string(name)?)
        }
    })
    .transpose()
}

pub(super) fn run(
    observable: Observable,
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    settings: Settings<'_>,
) -> Result<()> {
    use Observable::*;

    let Settings {
        mut pbc,
        tda,
        bfield,
        e_step,
        b_step,
        m1_basis,
        print_matrices,
    } = settings;
    if matches!(
        observable,
        Piezoelectric | Dielectric | BerryPolarization | Electromechanical
    ) {
        pbc.kmesh.gamma_centered = true;
        pbc.kmesh.fold_time_reversal = false;
    }
    let berry = BerryPolarizationOptions::from_kmesh(pbc.kmesh);
    let data = match observable {
        Magnetizability | MagneticPolarizability | CottonMouton | StaticMcd => {
            if system.lattice.is_some() {
                return Err(Gfn1Error::InvalidInput(
                    "static magnetic tensors require a molecule".into(),
                ));
            }
            if bfield.is_some() && matches!(observable, Magnetizability | CottonMouton) {
                return Err(Gfn1Error::InvalidInput(
                    "magnetizability and Cotton-Mouton use a zero reference magnetic field".into(),
                ));
            }
            let mut options = options.clone();
            let field = bfield.unwrap_or_default();
            options.external_field.magnetic_field = Some(field);
            let basis = load_m1_basis(m1_basis)?;
            let secondary = basis.as_ref();
            match observable {
                Magnetizability => {
                    let step = b_step.unwrap_or(0.01);
                    let xi = magnetic::magnetizability_tensor_analytic(
                        system, params, &options, secondary, step,
                    )?;
                    json!({"magnetizability_au": xi, "magnetic_step_au": step})
                }
                MagneticPolarizability => {
                    let alpha = magnetic::magnetic_polarizability(
                        system, params, &options, secondary, e_step,
                    )?;
                    json!({"magnetic_polarizability_au": alpha, "magnetic_field_au": field.to_array(),
                           "electric_step_au": e_step})
                }
                CottonMouton => {
                    let step = b_step.unwrap_or(0.02);
                    let tensor = magnetic::cotton_mouton_tensor(
                        system, params, &options, secondary, e_step, step,
                    )?;
                    json!({"cotton_mouton_au": tensor, "definition": "d^2 alpha_ij / dB_k^2 at B=0",
                           "axis_order": ["k","i","j"], "electric_step_au": e_step, "magnetic_step_au": step})
                }
                StaticMcd => {
                    let step = b_step.unwrap_or(0.01);
                    let tensor = magnetic::mcd_tensor(
                        system, params, &options, secondary, field, e_step, step,
                    )?;
                    json!({"static_dalpha_db_au": tensor, "definition": "static d alpha_ij / dB_k",
                           "axis_order": ["k","i","j"], "magnetic_field_au": field.to_array(),
                           "electric_step_au": e_step, "magnetic_step_au": step})
                }
                _ => unreachable!(),
            }
        }
        Ecd | OpticalRotation(_) => {
            if system.lattice.is_some() {
                return Err(Gfn1Error::InvalidInput(
                    "ECD and optical rotation require a molecule".into(),
                ));
            }
            let electronic = crate::run_electronic(system, params, options.clone())?;
            if let OpticalRotation(frequency) = observable {
                if !frequency.is_finite() || frequency < 0.0 {
                    return Err(Gfn1Error::InvalidInput(
                        "optical rotation requires a nonnegative photon energy in Hartree".into(),
                    ));
                }
                let beta = crate::tda_optical_rotation(
                    system,
                    params,
                    &electronic,
                    tda,
                    Vec3::zero(),
                    &[frequency],
                )?[0];
                if !beta.is_finite() {
                    return Err(Gfn1Error::InvalidInput(
                        "photon energy lies at an undamped optical pole".into(),
                    ));
                }
                json!({"photon_energy_hartree": frequency, "optical_rotation_beta_au": beta, "maximum_roots": tda.n_states})
            } else {
                let states =
                    crate::tda_rotatory_strengths(system, params, &electronic, tda, Vec3::zero())?;
                json!({"ecd_states": states.iter().map(|s| json!({
                    "excitation_energy_hartree": s.excitation_energy,
                    "rotatory_strength_au": s.rotatory_strength,
                    "magnetic_transition_dipole_au": s.magnetic_transition_dipole.to_array()
                })).collect::<Vec<_>>()})
            }
        }
        Piezoelectric => crate::pbc::polarization::response::pbc_polarization_derivatives(
            system, params, options, &pbc, &berry,
        )?
        .as_json(),
        Dielectric => crate::pbc::polarization::response::pbc_electronic_dielectric(
            system, params, options, &pbc, &berry,
        )?
        .as_json(),
        BerryPolarization => {
            let result = crate::pbc_berry_polarization(system, params, options, &pbc, &berry)?;
            json!({"polarization_au": result.polarization, "polarization_quantum_au": result.quantum,
                   "volume_bohr3": result.volume, "minimum_band_gap_hartree": result.min_band_gap})
        }
        Electromechanical => {
            let masses: Vec<_> = system
                .atoms
                .iter()
                .map(|a| crate::data_tables::relative_atomic_mass(a.z))
                .collect();
            let settings = crate::pbc::electromechanical::ElectromechanicalOptions {
                berry,
                ..Default::default()
            };
            crate::pbc::electromechanical::pbc_electromechanical_response(
                system, params, options, &pbc, &masses, &settings,
            )?
            .as_json()
        }
        Gruneisen(second_order) => {
            let settings = crate::GruneisenOptions {
                electronic: options.clone(),
                kpoint: !pbc.kmesh.is_gamma_only(),
                pbc,
                second_order,
                ..Default::default()
            };
            let result = crate::pbc_gruneisen(system, params, &settings)?;
            json!({"volume_bohr3": result.volume, "volume_step": result.delta,
                   "volume_step_second": result.delta_second, "frequencies_cm1": result.frequencies_cm1,
                   "mode_gamma": result.mode_gamma, "mode_gamma2": result.mode_gamma2,
                   "thermodynamic_gamma_at_kelvin": result.thermodynamic_gamma,
                   "thermodynamic_gamma2_at_kelvin": result.thermodynamic_gamma2,
                   "thermodynamic_gamma2_full_at_kelvin": result.thermodynamic_gamma2_full})
        }
        StrainHessian(step) => {
            let derivative = if pbc.kmesh.is_gamma_only() {
                crate::pbc_strain_hessian_derivative(system, params, options, &pbc, step)?
            } else {
                crate::pbc_kpoint_strain_hessian_derivative(system, params, options, &pbc, step)?
            };
            json!({"d_hessian_d_volume_strain_hartree_per_bohr2": rows(&derivative), "volume_step": step})
        }
        Vibrations => {
            let h = if system.lattice.is_some() {
                if pbc.kmesh.is_gamma_only() {
                    crate::pbc_gamma_hessian(system, params, options, &pbc)?.hessian
                } else {
                    crate::pbc_kpoint_hessian(system, params, options, &pbc)?.hessian
                }
            } else {
                crate::analytic_hessian(
                    system,
                    params,
                    crate::AnalyticHessianOptions {
                        electronic_options: options.clone(),
                        ..Default::default()
                    },
                )?
                .hessian
            };
            let numbers: Vec<_> = system.atoms.iter().map(|a| a.z).collect();
            let modes = if system.lattice.is_some() {
                crate::vibrational::vibrational_analysis(&h, &numbers)?
            } else {
                let positions: Vec<_> = system.atoms.iter().map(|a| a.position).collect();
                let masses: Vec<_> = numbers
                    .iter()
                    .map(|&z| crate::data_tables::relative_atomic_mass(z))
                    .collect();
                crate::molecular_vibrational_analysis(&h, &positions, &masses, Default::default())?
                    .vibrations
            };
            if print_matrices {
                json!({"wavenumbers_cm1": modes.wavenumbers, "cartesian_modes": modes.modes})
            } else {
                json!({"wavenumbers_cm1": modes.wavenumbers})
            }
        }
        Fragments { path, divide } => fragments(&path, divide, system, params, options, &pbc)?,
    };
    println!("{data:#}");
    Ok(())
}

fn rows(matrix: &crate::linalg::Matrix) -> Vec<Vec<f64>> {
    (0..matrix.rows())
        .map(|i| (0..matrix.cols()).map(|j| matrix[(i, j)]).collect())
        .collect()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Cut {
    bda: usize,
    baa: usize,
    #[serde(default)]
    image: [i32; 3],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FragmentInput {
    fragments: Vec<Vec<usize>>,
    fragment_charges: Option<Vec<f64>>,
    cuts: Option<Vec<Cut>>,
    buffers: Option<Vec<Vec<usize>>>,
    order: Option<usize>,
    compute_derivatives: Option<bool>,
    max_ao: Option<usize>,
}

fn fragments(
    path: &str,
    divide: bool,
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
) -> Result<serde_json::Value> {
    let input: FragmentInput = serde_json::from_str(&std::fs::read_to_string(path)?)
        .map_err(|e| Gfn1Error::InvalidInput(format!("invalid fragment input {path}: {e}")))?;
    if (divide && (input.cuts.is_some() || input.order.is_some()))
        || (!divide && input.buffers.is_some())
    {
        return Err(Gfn1Error::InvalidInput(
            "FMO takes cuts and order; divide-and-conquer takes buffers".into(),
        ));
    }
    let partition = FragmentPartition {
        charges: input
            .fragment_charges
            .unwrap_or_else(|| vec![0.; input.fragments.len()]),
        atoms: input.fragments,
        cuts: input
            .cuts
            .unwrap_or_default()
            .into_iter()
            .map(|c| CovalentCut {
                bda: c.bda,
                baa: c.baa,
                image: c.image,
            })
            .collect(),
    };
    let compute_derivatives = input.compute_derivatives.unwrap_or(true);
    let result = if divide {
        let mut settings = DivideConquerOptions {
            buffers: input.buffers.unwrap_or_default(),
            compute_derivatives,
            ..Default::default()
        };
        settings.max_ao = input.max_ao.unwrap_or(settings.max_ao);
        fragment::divide_and_conquer(system, params, options, pbc, &partition, &settings)?
    } else {
        let mut settings = FmoOptions {
            compute_derivatives,
            ..Default::default()
        };
        settings.max_ao = input.max_ao.unwrap_or(settings.max_ao);
        settings.order = input.order.unwrap_or(settings.order);
        fragment::fmo_xtb(system, params, options, pbc, &partition, &settings)?
    };
    Ok(json!({
        "total_free_hartree": result.total_free,
        "gradient_hartree_per_bohr": compute_derivatives.then(|| result.gradient.iter().map(|g| g.to_array()).collect::<Vec<_>>()),
        "forces_hartree_per_bohr": compute_derivatives.then(|| result.forces.iter().map(|g| g.to_array()).collect::<Vec<_>>()),
        "stress_hartree_per_bohr3": result.stress.as_ref().map(rows),
        "shell_charges_e": result.shell_charges, "embedding_iterations": result.embedding_iterations,
        "local_scc_iterations": result.local_scc_iterations, "cluster_count": result.cluster_count,
        "largest_cluster_ao": result.largest_cluster_ao
    }))
}
