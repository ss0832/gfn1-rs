// SPDX-License-Identifier: GPL-3.0-or-later

use super::observables::{self, Observable};
use crate::math::Vec3;
use crate::{
    active_targets_for_system, analytic_gradient, analytic_hessian, ir_spectrum, optimize_geometry,
    parameter_finite_difference, pbc_stress, raman_spectrum, run_electronic, solve_tda,
    solve_tda_gradient_method, static_polarizability, AnalyticGradientOptions,
    AnalyticHessianOptions, ElectronicOptions, GeometryOptimizationOptions, Gfn1Parameters,
    ParamDerivativeOptions, ParameterTarget, PbcOptions, PeriodicSystem, Result, SccAccelerator,
    TdaGradientMethod, TdaOptions, TdaSpin,
};

pub(super) fn run(arguments: Vec<String>) -> Result<()> {
    let mut args = arguments.into_iter();
    let mut xyz = None;
    let mut param = None;
    let mut d3_reference = None;
    let mut experimental_d4 = false;
    let mut d4_cutoff: Option<f64> = None;
    let mut d4_cn_cutoff: Option<f64> = None;
    let mut d4_atm = true;
    let mut d4_atm_cutoff: Option<f64> = None;
    let mut d4_s9: Option<f64> = None;
    let mut charge = None;
    let mut multiplicity = None;
    let mut bohr = false;
    let mut no_dispersion = false;
    let mut no_broyden = false;
    let mut mixing = None;
    let mut broyden_size = None;
    let mut electronic_temperature = None;
    let mut max_scc = None;
    let mut print_gradient = false;
    let mut print_hessian = false;
    let mut print_third = false;
    let mut print_fourth = false;
    let mut cubic_file = None;
    let mut quartic_file = None;
    let mut print_stress = false;
    let mut print_charges = false;
    let mut print_matrices = false;
    let mut optimize = false;
    let mut opt_max_iter = None;
    let mut opt_gtol = None;
    let mut opt_output = None;
    let mut opt_traj = None;
    let mut param_deriv = false;
    let mut param_targets: Option<String> = None;
    let mut all_param_targets = false;
    let mut param_forces = false;
    let mut param_stress = false;
    let mut param_step = 1.0e-4;
    let mut target_chunk: Option<String> = None;
    let mut field: Option<[f64; 3]> = None;
    let mut bfield: Option<[f64; 3]> = None;
    let mut print_polarizability = false;
    let mut print_ir = false;
    let mut print_raman = false;

    let mut field_step = None;
    let mut magnetic_step = None;
    let mut m1_basis = None;
    let mut observable = None;
    let mut level_shift = 0.0;
    let mut spinpol = false;
    let mut multipole = false;
    let mut multipole_octupole = false;
    let mut field_multipole = false;
    let mut multipole_third_order = false;
    let mut multipole_order = 0usize;
    let mut multipole_charge_order: Vec<usize> = Vec::new();
    let mut lr_exchange = false;
    let mut onsite_exchange = false;
    let mut dynamic_omega = false;
    let mut scf_trah = false;
    let mut multipole_secondary_basis: Option<String> = None;
    let mut camm_on_mdftb2 = false;
    let mut camm_damp = 1.0_f64;
    let mut camm_aes_scale = 1.0_f64;
    let mut camm_onsite_scale = 1.0_f64;
    let mut camm_damp_elem: Vec<(u8, f64)> = Vec::new();
    let mut camm_onsite_scale_elem: Vec<(u8, f64)> = Vec::new();
    let mut plus_u = false;
    let mut hubbard_u: Vec<(u8, f64)> = Vec::new();
    let mut plus_u_v = false;
    let mut hubbard_v: Vec<(u8, u8, f64)> = Vec::new();
    let mut hubbard_v_cutoff: Option<f64> = None;
    let mut linear_response_u = false;
    let mut plus_u_all_d = false;
    let mut camm_damp_charge: Option<(f64, f64)> = None;
    let mut camm_preset: Option<String> = None;
    // Track which CAMM knobs the user set explicitly, so a `--camm-preset` only fills the rest.
    let (mut camm_damp_set, mut camm_aes_set, mut camm_onsite_set) = (false, false, false);
    let mut charge_order = 3usize;
    let mut scc_accel: Option<String> = None;

    let mut print_tda = false;
    let mut print_rpa = false;
    let mut print_td_dfpt = false;
    let mut excited_hessian = false;
    let mut tda_manifold_average = false;
    let mut kmesh: Option<[usize; 3]> = None;
    let mut gamma_centered = false;
    let mut dfpt = false;
    let mut stda_gradient = false;
    let mut stda_hessian = false;
    let mut print_stda = false;
    let mut stda_options = crate::stda_xtb::NativeStdaOptions::default();
    let mut tda_n_states = 5usize;
    let mut tda_spin = "singlet".to_string();
    let mut tda_grad = false;
    let mut tda_state = 0usize;
    let mut tda_step = 1.0e-3;
    let mut tda_grad_method = "analytic".to_string();
    let mut nmr_nucleus: Option<usize> = None;
    let mut print_citations = false;
    let mut solvation = None;
    let mut solvent_state = crate::SolutionState::Gsolv;
    let mut solvent_temperature = 298.15;
    let mut solvent_grid = 230;
    let mut ml_model_path = None;
    let mut pbc_ao_cutoff = None;
    let mut soc_zeta_path = None;
    let mut print_vpt2 = false;
    let mut phonon_q: Option<[f64; 3]> = None;
    let mut phonon_repetitions = [1usize; 3];
    let mut log_file = None;
    let mut log_level = crate::LogLevel::Info;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--version" | "-V" => {
                println!("gfn1-rs {}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            "--log-file" => {
                log_file = Some(args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput("--log-file requires a path".into())
                })?)
            }
            "--log-level" => {
                log_level = crate::LogLevel::parse(&args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput("--log-level requires a level".into())
                })?)?
            }
            "--phonon-q" => {
                phonon_q = Some(parse_three(
                    args.next().ok_or_else(|| {
                        crate::Gfn1Error::InvalidInput("--phonon-q requires qx,qy,qz".into())
                    })?,
                    "--phonon-q",
                )?)
            }
            "--phonon-supercell" => {
                phonon_repetitions = parse_three(
                    args.next().ok_or_else(|| {
                        crate::Gfn1Error::InvalidInput(
                            "--phonon-supercell requires nx,ny,nz".into(),
                        )
                    })?,
                    "--phonon-supercell",
                )?
            }

            "--vpt2-pes" => print_vpt2 = true,
            "--magnetizability"
            | "--magnetic-polarizability"
            | "--cotton-mouton"
            | "--mcd"
            | "--ecd"
            | "--rotatory-strengths"
            | "--optical-rotation"
            | "--piezoelectric"
            | "--born-charges"
            | "--dielectric"
            | "--berry-polarization"
            | "--electromechanical"
            | "--gruneisen"
            | "--gruneisen-second-order"
            | "--strain-hessian-derivative"
            | "--frequencies"
            | "--fmo-fragments"
            | "--dc-fragments" => {
                if observable.is_some() {
                    return Err(crate::Gfn1Error::InvalidInput(
                        "choose one property flag per invocation".into(),
                    ));
                }
                observable = Some(match arg.as_str() {
                    "--magnetizability" => Observable::Magnetizability,
                    "--magnetic-polarizability" => Observable::MagneticPolarizability,
                    "--cotton-mouton" => Observable::CottonMouton,
                    "--mcd" => Observable::StaticMcd,
                    "--ecd" | "--rotatory-strengths" => Observable::Ecd,
                    "--optical-rotation" => {
                        Observable::OpticalRotation(parse_number(&mut args, &arg)?)
                    }
                    "--piezoelectric" | "--born-charges" => Observable::Piezoelectric,
                    "--dielectric" => Observable::Dielectric,
                    "--berry-polarization" => Observable::BerryPolarization,
                    "--electromechanical" => Observable::Electromechanical,
                    "--gruneisen" | "--gruneisen-second-order" => {
                        Observable::Gruneisen(arg == "--gruneisen-second-order")
                    }
                    "--strain-hessian-derivative" => {
                        Observable::StrainHessian(parse_number(&mut args, &arg)?)
                    }
                    "--frequencies" => Observable::Vibrations,
                    _ => Observable::Fragments {
                        path: required(&mut args, &arg)?,
                        divide: arg == "--dc-fragments",
                    },
                });
            }
            "--magnetic-field-step" | "--b-step" => {
                magnetic_step = Some(parse_number(&mut args, &arg)?)
            }
            "--m1-basis" => m1_basis = Some(required(&mut args, &arg)?),
            "--dfpt" => {
                dfpt = true;
                print_hessian = true;
            }
            "--kmesh" | "--kgrid" => {
                kmesh = Some(parse_three(required(&mut args, &arg)?, &arg)?);
            }
            "--gamma-centered" => gamma_centered = true,
            "--soc" | "--soc-zeta-file" => {
                soc_zeta_path = Some(args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput("--soc-zeta-file requires a path".into())
                })?)
            }

            "--list-ml-models" => {
                for name in crate::ml::MlHamiltonian::builtin_model_names() {
                    println!("{name}");
                }
                return Ok(());
            }
            "--ml-model" | "--ml-xtb" => {
                ml_model_path = Some(args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput(
                        "--ml-model requires a model path or builtin model specifier".into(),
                    )
                })?)
            }
            "--pbc-ao-cutoff" => {
                let cutoff = args
                    .next()
                    .ok_or_else(|| {
                        crate::Gfn1Error::InvalidInput(
                            "--pbc-ao-cutoff requires a positive bohr value".into(),
                        )
                    })?
                    .parse::<f64>()
                    .map_err(|_| {
                        crate::Gfn1Error::InvalidInput("invalid --pbc-ao-cutoff".into())
                    })?;
                if !cutoff.is_finite() || cutoff <= 0. {
                    return Err(crate::Gfn1Error::InvalidInput(
                        "--pbc-ao-cutoff must be positive and finite".into(),
                    ));
                }
                pbc_ao_cutoff = Some(cutoff);
            }
            "--alpb" | "--gbsa" => {
                let solvent = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput(format!("{arg} requires a solvent"))
                })?;
                if solvation.is_some() {
                    return Err(crate::Gfn1Error::InvalidInput(
                        "choose one solvation model".into(),
                    ));
                }
                solvation = Some(crate::SolvationOptions::new(
                    if arg == "--alpb" {
                        crate::SolvationModel::Alpb
                    } else {
                        crate::SolvationModel::Gbsa
                    },
                    solvent,
                ));
            }
            "--solvent-state" => {
                solvent_state = match args.next().as_deref() {
                    Some("gsolv") => crate::SolutionState::Gsolv,
                    Some("bar1mol") => crate::SolutionState::Bar1Mol,
                    Some("reference") => crate::SolutionState::Reference,
                    _ => {
                        return Err(crate::Gfn1Error::InvalidInput(
                            "--solvent-state requires gsolv | bar1mol | reference".into(),
                        ))
                    }
                };
            }
            "--solvent-temperature" => {
                solvent_temperature =
                    args.next().and_then(|x| x.parse().ok()).ok_or_else(|| {
                        crate::Gfn1Error::InvalidInput("invalid solvent temperature".into())
                    })?;
            }
            "--solvent-grid" => {
                solvent_grid = args
                    .next()
                    .and_then(|x| x.parse().ok())
                    .ok_or_else(|| crate::Gfn1Error::InvalidInput("invalid solvent grid".into()))?;
            }
            "--citations" => print_citations = true,
            "--param" => param = args.next(),
            "--d3-reference" => d3_reference = args.next(),
            "--experimental-d4" | "--d4" => experimental_d4 = true,
            "--d4-cutoff" => {
                let value = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput("--d4-cutoff needs a value".to_string())
                })?;
                d4_cutoff = Some(value.parse::<f64>().map_err(|_| {
                    crate::Gfn1Error::InvalidInput(format!("invalid --d4-cutoff `{value}`"))
                })?);
            }
            "--d4-cn-cutoff" => {
                let value = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput("--d4-cn-cutoff needs a value".to_string())
                })?;
                d4_cn_cutoff = Some(value.parse::<f64>().map_err(|_| {
                    crate::Gfn1Error::InvalidInput(format!("invalid --d4-cn-cutoff `{value}`"))
                })?);
            }
            "--no-d4-atm" | "--no-d4-threebody" | "--no-d4-three-body" | "--no-d4-3body" => {
                d4_atm = false;
            }
            "--d4-atm-cutoff" | "--d4-threebody-cutoff" | "--d4-3body-cutoff" => {
                let value = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput("--d4-atm-cutoff needs a value".to_string())
                })?;
                d4_atm_cutoff = Some(value.parse::<f64>().map_err(|_| {
                    crate::Gfn1Error::InvalidInput(format!("invalid --d4-atm-cutoff `{value}`"))
                })?);
            }
            "--d4-s9" => {
                let value = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput("--d4-s9 needs a value".to_string())
                })?;
                d4_s9 = Some(value.parse::<f64>().map_err(|_| {
                    crate::Gfn1Error::InvalidInput(format!("invalid --d4-s9 `{value}`"))
                })?);
            }
            "--no-dispersion" => no_dispersion = true,
            "--multipole" => multipole = true,
            "--multipole-octupole" => {
                multipole = true;
                multipole_octupole = true;
            }
            "--field-multipole" => {
                multipole = true;
                field_multipole = true;
            }
            "--multipole-third-order" => {
                multipole = true;
                multipole_third_order = true;
            }
            "--multipole-secondary-basis" => {
                multipole = true;
                multipole_secondary_basis = args.next();
            }
            "--multipole-model" => {
                let value = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput(
                        "--multipole-model needs a value (mdftb2 | camm_on_mdftb2)".to_string(),
                    )
                })?;
                match value.as_str() {
                    "mdftb2" => {}
                    "camm_on_mdftb2" | "camm-on-mdftb2" | "camm" => {
                        multipole = true;
                        camm_on_mdftb2 = true;
                    }
                    other => {
                        return Err(crate::Gfn1Error::InvalidInput(format!(
                            "--multipole-model `{other}` (want mdftb2 | camm_on_mdftb2)"
                        )))
                    }
                }
            }
            "--camm-damp" => {
                multipole = true;
                camm_on_mdftb2 = true;
                let value = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput(
                        "--camm-damp needs the range factor κ (> 0, e.g. 1.0)".to_string(),
                    )
                })?;
                camm_damp = value.parse::<f64>().map_err(|_| {
                    crate::Gfn1Error::InvalidInput(format!("invalid --camm-damp `{value}`"))
                })?;
                camm_damp_set = true;
            }
            "--camm-aes-scale" => {
                multipole = true;
                camm_on_mdftb2 = true;
                let value = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput(
                        "--camm-aes-scale needs the amplitude s_AES (≥ 0, e.g. 1.0)".to_string(),
                    )
                })?;
                camm_aes_scale = value.parse::<f64>().map_err(|_| {
                    crate::Gfn1Error::InvalidInput(format!("invalid --camm-aes-scale `{value}`"))
                })?;
                camm_aes_set = true;
            }
            "--camm-preset" => {
                multipole = true;
                camm_on_mdftb2 = true;
                let value = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput(
                        "--camm-preset needs a preset name (polar | halogen | halogen-v1 | halogen-allgrad | sigma-hole)".to_string(),
                    )
                })?;
                camm_preset = Some(value);
            }
            "--camm-onsite-scale" => {
                multipole = true;
                camm_on_mdftb2 = true;
                let value = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput(
                        "--camm-onsite-scale needs the on-site penalty scale s_onsite (≥ 0, e.g. 1.0)"
                            .to_string(),
                    )
                })?;
                camm_onsite_scale = value.parse::<f64>().map_err(|_| {
                    crate::Gfn1Error::InvalidInput(format!("invalid --camm-onsite-scale `{value}`"))
                })?;
                camm_onsite_set = true;
            }
            "--camm-damp-elem" => {
                multipole = true;
                camm_on_mdftb2 = true;
                let value = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput(
                        "--camm-damp-elem needs per-element κ, e.g. `N:3.0,O:3.0,C:1.5`"
                            .to_string(),
                    )
                })?;
                for tok in value.split(',').filter(|t| !t.trim().is_empty()) {
                    let (sym, kstr) = tok.split_once(':').ok_or_else(|| {
                        crate::Gfn1Error::InvalidInput(format!(
                            "invalid --camm-damp-elem entry `{tok}` (want `Elem:κ`)"
                        ))
                    })?;
                    let z = sym
                        .trim()
                        .parse::<u8>()
                        .ok()
                        .or_else(|| crate::symbol_to_z(sym.trim()))
                        .ok_or_else(|| {
                            crate::Gfn1Error::InvalidInput(format!(
                                "invalid element `{sym}` in --camm-damp-elem"
                            ))
                        })?;
                    let k = kstr.trim().parse::<f64>().map_err(|_| {
                        crate::Gfn1Error::InvalidInput(format!(
                            "invalid κ `{kstr}` in --camm-damp-elem"
                        ))
                    })?;
                    camm_damp_elem.push((z, k));
                }
            }
            "--camm-onsite-scale-elem" => {
                multipole = true;
                camm_on_mdftb2 = true;
                let value = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput(
                        "--camm-onsite-scale-elem needs per-element s_onsite, e.g. `Cl:0.02,Si:1.0`"
                            .to_string(),
                    )
                })?;
                for tok in value.split(',').filter(|t| !t.trim().is_empty()) {
                    let (sym, sstr) = tok.split_once(':').ok_or_else(|| {
                        crate::Gfn1Error::InvalidInput(format!(
                            "invalid --camm-onsite-scale-elem entry `{tok}` (want `Elem:s_onsite`)"
                        ))
                    })?;
                    let z = sym
                        .trim()
                        .parse::<u8>()
                        .ok()
                        .or_else(|| crate::symbol_to_z(sym.trim()))
                        .ok_or_else(|| {
                            crate::Gfn1Error::InvalidInput(format!(
                                "invalid element `{sym}` in --camm-onsite-scale-elem"
                            ))
                        })?;
                    let s = sstr.trim().parse::<f64>().map_err(|_| {
                        crate::Gfn1Error::InvalidInput(format!(
                            "invalid s_onsite `{sstr}` in --camm-onsite-scale-elem"
                        ))
                    })?;
                    camm_onsite_scale_elem.push((z, s));
                }
            }
            "--hubbard-u" => {
                plus_u = true;
                let value = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput(
                        "--hubbard-u needs per-element U (Hartree), e.g. `Fe:0.15,Ni:0.12`"
                            .to_string(),
                    )
                })?;
                for tok in value.split(',').filter(|t| !t.trim().is_empty()) {
                    let (sym, ustr) = tok.split_once(':').ok_or_else(|| {
                        crate::Gfn1Error::InvalidInput(format!(
                            "invalid --hubbard-u entry `{tok}` (want `Elem:U`)"
                        ))
                    })?;
                    let z = sym
                        .trim()
                        .parse::<u8>()
                        .ok()
                        .or_else(|| crate::symbol_to_z(sym.trim()))
                        .ok_or_else(|| {
                            crate::Gfn1Error::InvalidInput(format!(
                                "invalid element `{sym}` in --hubbard-u"
                            ))
                        })?;
                    let u = ustr.trim().parse::<f64>().map_err(|_| {
                        crate::Gfn1Error::InvalidInput(format!("invalid U `{ustr}` in --hubbard-u"))
                    })?;
                    hubbard_u.push((z, u));
                }
            }
            "--hubbard-v" => {
                plus_u = true;
                plus_u_v = true;
                let value = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput(
                        "--hubbard-v needs element-pair V, e.g. `Fe:N:0.04`".to_string(),
                    )
                })?;
                for tok in value.split(',').filter(|t| !t.trim().is_empty()) {
                    let parts: Vec<&str> = tok.split(':').collect();
                    if parts.len() != 3 {
                        return Err(crate::Gfn1Error::InvalidInput(format!(
                            "invalid --hubbard-v entry `{tok}` (want `Elem:Elem:V`)"
                        )));
                    }
                    let zof = |sym: &str| {
                        sym.trim()
                            .parse::<u8>()
                            .ok()
                            .or_else(|| crate::symbol_to_z(sym.trim()))
                    };
                    let za = zof(parts[0]).ok_or_else(|| {
                        crate::Gfn1Error::InvalidInput(format!(
                            "invalid element `{}` in --hubbard-v",
                            parts[0]
                        ))
                    })?;
                    let zb = zof(parts[1]).ok_or_else(|| {
                        crate::Gfn1Error::InvalidInput(format!(
                            "invalid element `{}` in --hubbard-v",
                            parts[1]
                        ))
                    })?;
                    let v = parts[2].trim().parse::<f64>().map_err(|_| {
                        crate::Gfn1Error::InvalidInput(format!(
                            "invalid V `{}` in --hubbard-v",
                            parts[2]
                        ))
                    })?;
                    hubbard_v.push((za, zb, v));
                }
            }
            "--hubbard-v-cutoff" => {
                let value = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput(
                        "--hubbard-v-cutoff needs a distance (bohr)".to_string(),
                    )
                })?;
                hubbard_v_cutoff = Some(value.trim().parse::<f64>().map_err(|_| {
                    crate::Gfn1Error::InvalidInput(format!("invalid --hubbard-v-cutoff `{value}`"))
                })?);
            }
            "--camm-damp-charge" => {
                multipole = true;
                camm_on_mdftb2 = true;
                let value = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput(
                        "--camm-damp-charge needs `κ0,γ`, e.g. `3.0,4.0` (κ_A = κ0/(1+γ·Δq_A²))"
                            .to_string(),
                    )
                })?;
                let (k0s, gs) = value.split_once(',').ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput(format!(
                        "invalid --camm-damp-charge `{value}` (want `κ0,γ`)"
                    ))
                })?;
                let k0 = k0s
                    .trim()
                    .parse::<f64>()
                    .map_err(|_| crate::Gfn1Error::InvalidInput(format!("invalid κ0 `{k0s}`")))?;
                let g = gs
                    .trim()
                    .parse::<f64>()
                    .map_err(|_| crate::Gfn1Error::InvalidInput(format!("invalid γ `{gs}`")))?;
                camm_damp_charge = Some((k0, g));
            }
            "--multipole-order" => {
                multipole = true;
                let value = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput(
                        "--multipole-order needs an integer rank (>=4 enables the experimental \
                         arbitrary-rank path; <4 ≡ legacy dipole/quad[/octupole])"
                            .to_string(),
                    )
                })?;
                multipole_order = value.parse::<usize>().map_err(|_| {
                    crate::Gfn1Error::InvalidInput(format!("invalid --multipole-order `{value}`"))
                })?;
            }
            "--multipole-charge-order" => {
                multipole = true;
                let value = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput(
                        "--multipole-charge-order needs a comma-separated per-rank list, e.g. \
                         `6,4,2,2` (dipole→6th, quadrupole→4th, octupole/hexadecapole→off). Entry \
                         l is the max on-site charge order coupled to the 2^l-pole; it must satisfy \
                         order ≤ 2l+3 (the rank-l self-energy terminates there). Requires \
                         --multipole-order ≥ the highest rank with a cross term."
                            .to_string(),
                    )
                })?;
                multipole_charge_order = value
                    .split(',')
                    .map(|t| t.trim())
                    .filter(|t| !t.is_empty())
                    .map(|t| {
                        t.parse::<usize>().map_err(|_| {
                            crate::Gfn1Error::InvalidInput(format!(
                                "invalid --multipole-charge-order entry `{t}` (want non-negative integers)"
                            ))
                        })
                    })
                    .collect::<std::result::Result<Vec<usize>, _>>()?;
            }
            "--lr-exchange" => lr_exchange = true,
            "--onsite-exchange" => {
                lr_exchange = true;
                onsite_exchange = true;
            }
            "--dynamic-omega" => {
                lr_exchange = true;
                dynamic_omega = true;
            }
            "--scf-trah" => scf_trah = true,
            "--charge-order" => {
                let value = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput(
                        "--charge-order needs an integer (>=3; 3 = stock GFN1)".to_string(),
                    )
                })?;
                charge_order = value.parse::<usize>().map_err(|_| {
                    crate::Gfn1Error::InvalidInput(format!("invalid --charge-order `{value}`"))
                })?;
            }
            "--no-broyden" => no_broyden = true,
            "--gradient" | "--grad" => print_gradient = true,
            "--hessian" | "--hess" => print_hessian = true,
            "--third-derivative" | "--cubic" => print_third = true,
            "--fourth-derivative" | "--quartic" => print_fourth = true,
            "--cubic-file" | "--quartic-file" => {
                let path = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput(format!("{arg} needs an output path"))
                })?;
                if arg == "--cubic-file" {
                    cubic_file = Some(std::path::PathBuf::from(path));
                    print_third = true;
                } else {
                    quartic_file = Some(std::path::PathBuf::from(path));
                    print_fourth = true;
                }
            }
            "--stress" => print_stress = true,
            "--optimize" | "--opt" => optimize = true,
            "--opt-max-iter" => {
                let value = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput("--opt-max-iter needs a value".to_string())
                })?;
                opt_max_iter = Some(value.parse::<usize>().map_err(|_| {
                    crate::Gfn1Error::InvalidInput(format!(
                        "invalid --opt-max-iter value `{value}`"
                    ))
                })?);
            }
            "--opt-gtol" => {
                let value = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput("--opt-gtol needs a value".to_string())
                })?;
                opt_gtol = Some(value.parse::<f64>().map_err(|_| {
                    crate::Gfn1Error::InvalidInput(format!("invalid --opt-gtol value `{value}`"))
                })?);
            }
            "--opt-output" => opt_output = args.next(),
            "--opt-traj" => opt_traj = args.next(),
            "--mixing" => {
                let value = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput("--mixing needs a value".to_string())
                })?;
                mixing = Some(value.parse::<f64>().map_err(|_| {
                    crate::Gfn1Error::InvalidInput(format!("invalid --mixing value `{value}`"))
                })?);
            }
            "--electronic-temperature" | "--etemp" => {
                let value = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput(
                        "--electronic-temperature needs a value (K)".to_string(),
                    )
                })?;
                electronic_temperature = Some(value.parse::<f64>().map_err(|_| {
                    crate::Gfn1Error::InvalidInput(format!(
                        "invalid --electronic-temperature value `{value}`"
                    ))
                })?);
            }
            "--max-scc" => {
                let value = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput("--max-scc needs a value".to_string())
                })?;
                max_scc = Some(value.parse::<usize>().map_err(|_| {
                    crate::Gfn1Error::InvalidInput(format!("invalid --max-scc value `{value}`"))
                })?);
            }
            "--broyden-size" => {
                let value = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput("--broyden-size needs a value".to_string())
                })?;
                broyden_size = Some(value.parse::<usize>().map_err(|_| {
                    crate::Gfn1Error::InvalidInput(format!(
                        "invalid --broyden-size value `{value}`"
                    ))
                })?);
            }
            "--charges" => print_charges = true,
            "--matrices" => print_matrices = true,
            "--charge" => {
                let value = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput("--charge needs a value".to_string())
                })?;
                charge = Some(value.parse::<f64>().map_err(|_| {
                    crate::Gfn1Error::InvalidInput(format!("invalid --charge value `{value}`"))
                })?);
            }
            "--multiplicity" | "--spin-multiplicity" => {
                let value = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput("--multiplicity needs a value".to_string())
                })?;
                multiplicity = Some(value.parse::<usize>().map_err(|_| {
                    crate::Gfn1Error::InvalidInput(format!(
                        "invalid --multiplicity value `{value}`"
                    ))
                })?);
            }
            "--bohr" => bohr = true,
            "--spinpol" | "--spin-polarization" => spinpol = true,
            "--plus-u" | "--dft-plus-u" => plus_u = true,
            "--plus-u-v" => {
                plus_u = true;
                plus_u_v = true;
            }
            "--linear-response-u" | "--hubbard-u-linear-response" => {
                plus_u = true;
                linear_response_u = true;
            }
            "--plus-u-all-d" => {
                plus_u = true;
                plus_u_all_d = true;
            }
            "--field" => {
                let mut values = [0.0f64; 3];
                for (axis, slot) in values.iter_mut().enumerate() {
                    let value = args.next().ok_or_else(|| {
                        crate::Gfn1Error::InvalidInput(format!(
                            "--field needs three components (Ex Ey Ez); missing component {}",
                            axis + 1
                        ))
                    })?;
                    *slot = value.parse::<f64>().map_err(|_| {
                        crate::Gfn1Error::InvalidInput(format!("invalid --field value `{value}`"))
                    })?;
                }
                field = Some(values);
            }
            "--bfield" => {
                let mut values = [0.0f64; 3];
                for (axis, slot) in values.iter_mut().enumerate() {
                    let value = args.next().ok_or_else(|| {
                        crate::Gfn1Error::InvalidInput(format!(
                            "--bfield needs three components (Bx By Bz); missing component {}",
                            axis + 1
                        ))
                    })?;
                    *slot = value.parse::<f64>().map_err(|_| {
                        crate::Gfn1Error::InvalidInput(format!("invalid --bfield value `{value}`"))
                    })?;
                }
                bfield = Some(values);
            }
            "--polarizability" | "--polar" => print_polarizability = true,
            "--ir" => print_ir = true,
            "--raman" => print_raman = true,

            "--tda" => print_tda = true,
            "--tda-manifold-average" | "--rpa-manifold-average" => tda_manifold_average = true,
            "--rpa" | "--full-td" => print_rpa = true,
            "--td-dfpt" => print_td_dfpt = true,
            "--stda-xtb" => print_stda = true,
            "--stda-thermal" => {
                print_stda = true;
                stda_options.thermal_response = Some(true);
            }
            "--stda-legacy" => {
                print_stda = true;
                stda_options.thermal_response = Some(false);
            }
            "--stda-manifold-average" => stda_options.manifold_average = true,
            "--stda-grad" => {
                print_stda = true;
                stda_gradient = true;
            }
            "--stda-hessian" => {
                print_stda = true;
                stda_hessian = true;
            }
            "--tda-hessian" | "--rpa-hessian" => {
                print_tda |= arg == "--tda-hessian";
                print_rpa |= arg == "--rpa-hessian";
                excited_hessian = true;
            }
            "--rpa-grad" => {
                print_rpa = true;
                tda_grad = true;
            }
            "--stda-triplet" => stda_options.triplet = true,
            "--stda-window-ev" => {
                stda_options.energy_window_ev =
                    args.next().and_then(|s| s.parse().ok()).ok_or_else(|| {
                        crate::Gfn1Error::InvalidInput(
                            "--stda-window-ev requires a positive energy in eV".into(),
                        )
                    })?;
            }
            "--tda-grad" => {
                print_tda = true;
                tda_grad = true;
            }
            "--tda-spin" | "--rpa-spin" => tda_spin = required(&mut args, &arg)?,
            "--tda-gradient-method" => tda_grad_method = required(&mut args, &arg)?,
            "--nmr" => {
                let value = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput(
                        "--nmr needs a nucleus index (0-based)".to_string(),
                    )
                })?;
                nmr_nucleus = Some(value.parse::<usize>().map_err(|_| {
                    crate::Gfn1Error::InvalidInput(format!("invalid --nmr value `{value}`"))
                })?);
            }
            "--tda-state" | "--rpa-state" | "--stda-state" => {
                let value = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput("--tda-state needs a value".to_string())
                })?;
                tda_state = value.parse::<usize>().map_err(|_| {
                    crate::Gfn1Error::InvalidInput(format!("invalid --tda-state value `{value}`"))
                })?;
            }
            "--tda-step" => {
                let value = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput("--tda-step needs a value".to_string())
                })?;
                tda_step = value.parse::<f64>().map_err(|_| {
                    crate::Gfn1Error::InvalidInput(format!("invalid --tda-step value `{value}`"))
                })?;
            }
            "--n-states" => {
                let value = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput("--n-states needs a value".to_string())
                })?;
                tda_n_states = value.parse::<usize>().map_err(|_| {
                    crate::Gfn1Error::InvalidInput(format!("invalid --n-states value `{value}`"))
                })?;
            }
            "--scc-accel" => scc_accel = args.next(),
            "--level-shift" => {
                let value = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput("--level-shift needs a value".to_string())
                })?;
                level_shift = value.parse::<f64>().map_err(|_| {
                    crate::Gfn1Error::InvalidInput(format!("invalid --level-shift value `{value}`"))
                })?;
            }

            "--field-step" => field_step = Some(parse_number(&mut args, &arg)?),
            "--param-deriv" => param_deriv = true,
            "--all-param-targets" => all_param_targets = true,
            "--param-forces" => param_forces = true,
            "--param-stress" => param_stress = true,
            "--target-chunk" => target_chunk = args.next(),
            "--targets" => param_targets = args.next(),
            "--param-step" => {
                let value = args.next().ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput("--param-step needs a value".to_string())
                })?;
                param_step = value.parse::<f64>().map_err(|_| {
                    crate::Gfn1Error::InvalidInput(format!("invalid --param-step value `{value}`"))
                })?;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other if xyz.is_none() && !other.starts_with('-') => xyz = Some(other.to_string()),
            other => {
                return Err(crate::Gfn1Error::InvalidInput(format!(
                    "unexpected argument `{other}`"
                )));
            }
        }
    }

    if print_citations && xyz.is_none() {
        let mut selected = ElectronicOptions::default();
        if let Some(spec) = &ml_model_path {
            selected.ml_model = Some(std::sync::Arc::new(crate::ml::MlHamiltonian::resolve(
                spec,
            )?));
        }
        selected.solvation = solvation;
        selected.spin_polarization = spinpol;
        selected.external_field.magnetic_field = bfield.map(|b| Vec3::new(b[0], b[1], b[2]));
        print!(
            "{}",
            crate::citations::bibliography(&crate::citations::electronic_citations(
                &selected, false
            ))?
        );
        return Ok(());
    }
    let xyz = xyz.ok_or_else(|| crate::Gfn1Error::InvalidInput("missing XYZ file".to_string()))?;

    if [print_tda, print_rpa, print_td_dfpt, print_stda]
        .into_iter()
        .filter(|&x| x)
        .count()
        > 1
    {
        return Err(crate::Gfn1Error::InvalidInput(
            "select one of TDA, RPA, TD-DFPT or sTDA-xTB".into(),
        ));
    }
    if print_td_dfpt
        && (print_gradient
            || print_hessian
            || print_third
            || print_fourth
            || optimize
            || print_vpt2
            || print_ir
            || print_raman
            || print_polarizability
            || print_stress
            || bfield.is_some()
            || nmr_nucleus.is_some()
            || phonon_q.is_some())
    {
        return Err(crate::Gfn1Error::InvalidInput(
            "--td-dfpt currently computes optical excitation energies".into(),
        ));
    }
    if (print_tda || print_rpa)
        && (dfpt
            || print_third
            || print_fourth
            || optimize
            || print_vpt2
            || print_ir
            || print_raman
            || print_polarizability
            || print_stress
            || bfield.is_some()
            || nmr_nucleus.is_some()
            || phonon_q.is_some())
    {
        return Err(crate::Gfn1Error::InvalidInput(
            "incompatible TD-GFN1 output options".into(),
        ));
    }
    if tda_n_states == 0 {
        return Err(crate::Gfn1Error::InvalidInput(
            "--n-states must be positive".into(),
        ));
    }

    if observable.is_some()
        && (print_tda
            || print_rpa
            || print_td_dfpt
            || print_stda
            || print_gradient
            || print_hessian
            || print_third
            || print_fourth
            || optimize
            || print_vpt2
            || print_ir
            || print_raman
            || print_polarizability
            || print_stress
            || nmr_nucleus.is_some()
            || phonon_q.is_some()
            || soc_zeta_path.is_some()
            || param_deriv)
    {
        return Err(crate::Gfn1Error::InvalidInput(
            "the requested property requires a separate invocation".into(),
        ));
    }
    let magnetic_observable = matches!(
        observable,
        Some(
            Observable::Magnetizability
                | Observable::MagneticPolarizability
                | Observable::CottonMouton
                | Observable::StaticMcd
        )
    );
    if observable.is_some() && !magnetic_observable && bfield.is_some() {
        return Err(crate::Gfn1Error::InvalidInput(
            "magnetic field is incompatible with the requested property".into(),
        ));
    }
    if (m1_basis.is_some() || magnetic_step.is_some())
        && !(magnetic_observable || bfield.is_some() || nmr_nucleus.is_some())
    {
        return Err(crate::Gfn1Error::InvalidInput(
            "--m1-basis and --magnetic-field-step require a magnetic calculation".into(),
        ));
    }
    if tda_manifold_average
        && !((print_tda || print_rpa) && (tda_grad || print_gradient || excited_hessian || print_hessian))
    {
        return Err(crate::Gfn1Error::InvalidInput(
            "manifold averaging requires a TDA or RPA gradient or Hessian".into(),
        ));
    }
    if (print_third || print_fourth)
        && (print_gradient
            || print_hessian
            || optimize
            || print_vpt2
            || print_polarizability
            || print_ir
            || print_raman
            || print_stress
            || bfield.is_some()
            || nmr_nucleus.is_some()
            || phonon_q.is_some()
            || param_deriv)
    {
        return Err(crate::Gfn1Error::InvalidInput(
            "FC3/FC4 must be requested separately from other calculation outputs".into(),
        ));
    }
    if stda_options.manifold_average
        && !(print_stda && (stda_gradient || stda_hessian || print_gradient || print_hessian))
    {
        return Err(crate::Gfn1Error::InvalidInput(
            "--stda-manifold-average requires an sTDA gradient or Hessian".into(),
        ));
    }
    if print_stda {
        if kmesh.is_some()
            || gamma_centered
            || soc_zeta_path.is_some()
            || param.is_some()
            || solvation.is_some()
            || ml_model_path.is_some()
            || multipole
            || lr_exchange
            || plus_u
            || experimental_d4
            || spinpol
            || field.is_some()
            || bfield.is_some()
            || dfpt
            || print_third
            || print_fourth
            || print_stress
            || optimize
            || print_vpt2
            || multiplicity.is_some()
        {
            return Err(crate::Gfn1Error::InvalidInput("--stda-xtb selects its dedicated native Rust model; incompatible GFN1 modifiers or derivative flags were supplied".into()));
        }
        let system = PeriodicSystem::from_xyz_file(&xyz, charge.unwrap_or(0.), bohr)?;
        stda_options.electronic_temperature = electronic_temperature.unwrap_or(300.);

        stda_options.n_states = tda_n_states;
        if stda_gradient || stda_hessian || print_gradient || print_hessian {
            let parameters = Gfn1Parameters::resolve(None)?;
            let electronic = ElectronicOptions {
                electronic_temperature: stda_options.electronic_temperature,
                enable_dispersion: !no_dispersion,
                ..Default::default()
            };
            println!("model GFN1 free energy + signed sTDA-xTB excitation");
            println!("stda_state {tda_state}");
            if stda_gradient || print_gradient {
                let result = crate::stda_xtb::solve_stda_xtb_total_gradient(
                    &system,
                    &parameters,
                    &electronic,
                    tda_state,
                    &stda_options,
                )?;
                print_stda_response_metadata(result.thermal_response, &result.roots, stda_options.manifold_average);
                println!("stda_total_energy {:.14}", result.total_energy);
                print_gradient_rows("stda_gradient_hartree_per_bohr", &result.gradient);
            }
            if stda_hessian || print_hessian {
                let result = crate::stda_xtb::solve_stda_xtb_total_hessian(
                    &system,
                    &parameters,
                    &electronic,
                    tda_state,
                    &stda_options,
                )?;
                print_stda_response_metadata(result.thermal_response, &result.roots, stda_options.manifold_average);
                println!("stda_total_energy {:.14}", result.total_energy);
                print_matrix("stda_hessian_hartree_per_bohr2", &result.hessian);
            }
            return Ok(());
        }
        let result = crate::stda_xtb::solve_stda_xtb(&system, &stda_options)?;
        println!("model sTDA-xTB (native Rust)");
        print_stda_response_metadata(result.thermal_response, &(0..0), false);
        println!(
            "electronic_temperature_kelvin {}",
            result.electronic_temperature_kelvin
        );
        println!(
            "maximum_occupation_deviation {:e}",
            result.maximum_occupation_deviation
        );
        println!(
            "parameter_source_commit {}",
            crate::stda_xtb::XTB4STDA_COMMIT
        );
        println!(
            "response_reference_commit {}",
            crate::stda_xtb::STD2_REFERENCE_COMMIT
        );
        println!("basis_size {}", result.basis_size);
        println!("configurations {}", result.configurations.len());
        println!("root excitation_ev oscillator_strength");
        for (i, state) in result.states.iter().enumerate() {
            println!(
                "{} {:.8} {:.8}",
                i + 1,
                state.excitation_ev,
                state.oscillator_strength
            );
        }
        return Ok(());
    }
    let params = Gfn1Parameters::resolve(param.as_deref())?;
    println!("parameters: {}", params.source_description());
    let system = PeriodicSystem::from_xyz_file(&xyz, charge.unwrap_or(0.0), bohr)?;

    if (kmesh.is_some() || dfpt) && system.lattice.is_none() {
        return Err(crate::Gfn1Error::InvalidInput(
            "--kmesh and --dfpt require a periodic input".into(),
        ));
    }
    if gamma_centered && kmesh.is_none() {
        return Err(crate::Gfn1Error::InvalidInput(
            "--gamma-centered requires --kmesh".into(),
        ));
    }
    if let Some(mesh) = kmesh {
        if mesh.contains(&0)
            || mesh
                .into_iter()
                .try_fold(1usize, usize::checked_mul)
                .is_none()
            || mesh
                .iter()
                .zip(system.lattice.as_ref().expect("validated lattice").periodic)
                .any(|(&n, pbc)| n > 1 && !pbc)
        {
            return Err(crate::Gfn1Error::InvalidInput(
                "--kmesh requires positive dimensions and 1 along nonperiodic axes".into(),
            ));
        }
        if optimize || print_fourth || phonon_q.is_some() {
            return Err(crate::Gfn1Error::InvalidInput(
                "--kmesh is unavailable for optimization, FC4 and phonon supercells in this CLI"
                    .into(),
            ));
        }
    }
    let periodic_options = |options: &ElectronicOptions| -> Result<PbcOptions> {
        let mut pbc = PbcOptions::for_electronic(options)?;
        if let Some(size) = kmesh {
            pbc.kmesh = crate::KMesh {
                gamma_centered,
                ..crate::KMesh::monkhorst_pack(size)
            };
        }
        Ok(pbc)
    };
    let mut options = ElectronicOptions::default();
    options.pbc_ao_cutoff = pbc_ao_cutoff;
    if let Some(path) = log_file {
        options.logger = Some(crate::CalculationLogger::append(path, log_level, true)?);
    }
    if let Some(path) = ml_model_path {
        let (model, identity) = crate::ml::MlHamiltonian::resolve_with_identity(&path)?;
        println!("ml_model {}", identity.description());
        if let Some(logger) = &options.logger {
            logger.event(crate::LogLevel::Info, "ml_model_loaded", identity.as_json())?;
        }
        options.ml_model = Some(std::sync::Arc::new(model));
    }
    if print_vpt2 || print_tda || print_rpa || print_td_dfpt {
        options.energy_tolerance = options.energy_tolerance.min(1e-12);
        options.charge_tolerance = options.charge_tolerance.min(1e-11);
    }
    if let Some(s) = &mut solvation {
        s.state = solvent_state;
        s.temperature = solvent_temperature;
        s.grid_points = solvent_grid;
    }
    options.solvation = solvation;
    options.charge = charge;
    options.spin_multiplicity = multiplicity;
    options.d3_reference_path = d3_reference;
    options.enable_dispersion = !no_dispersion;
    options.experimental_d4 = experimental_d4;
    if let Some(value) = d4_cutoff {
        options.d4_cutoff = value;
    }
    if let Some(value) = d4_cn_cutoff {
        options.d4_cn_cutoff = value;
    }
    options.d4_atm = d4_atm;
    if let Some(value) = d4_atm_cutoff {
        options.d4_atm_cutoff = value;
    }
    if let Some(value) = d4_s9 {
        options.d4_s9 = Some(value);
    }
    options.scc_broyden = !no_broyden;
    if let Some(mixing) = mixing {
        options.mixing = mixing;
    }
    if let Some(broyden_size) = broyden_size {
        options.scc_broyden_size = broyden_size;
    }
    if let Some(temp) = electronic_temperature {
        options.electronic_temperature = temp;
    }
    if let Some(scc) = max_scc {
        options.max_scc = scc;
    }
    if let Some([ex, ey, ez]) = field {
        options.external_field = crate::ExternalFieldOptions::electric(Vec3::new(ex, ey, ez));
    }
    options.level_shift = level_shift;
    options.spin_polarization = spinpol;
    options.plus_u = plus_u;
    options.hubbard_u = hubbard_u;
    options.plus_u_v = plus_u_v;
    options.hubbard_v = hubbard_v;
    if let Some(c) = hubbard_v_cutoff {
        options.hubbard_v_cutoff = c;
    }
    options.hubbard_u_linear_response = linear_response_u;
    options.plus_u_all_d = plus_u_all_d;
    options.multipole = multipole;
    options.multipole_octupole = multipole_octupole;
    options.field_multipole = field_multipole;
    options.multipole_third_order = multipole_third_order;
    options.multipole_order = multipole_order;
    options.multipole_charge_order = multipole_charge_order;
    options.multipole_model = if camm_on_mdftb2 {
        crate::MultipoleModel::CammOnMdftb2
    } else {
        crate::MultipoleModel::Mdftb2
    };
    // Resolve a named CAMM preset, filling only the knobs the user did not set explicitly
    // (explicit `--camm-damp*` / `--camm-aes-scale` / `--camm-onsite-scale` always win).
    if let Some(name) = &camm_preset {
        let (gk, elems, aes, onsite, onsite_elem) = crate::camm_preset(name).ok_or_else(|| {
            crate::Gfn1Error::InvalidInput(format!(
                "unknown --camm-preset `{name}` (valid: polar | halogen | halogen-v1 | halogen-allgrad | sigma-hole)"
            ))
        })?;
        if !camm_damp_set {
            camm_damp = gk;
        }
        if !camm_aes_set {
            camm_aes_scale = aes;
        }
        if !camm_onsite_set {
            camm_onsite_scale = onsite;
        }
        if camm_damp_elem.is_empty() {
            camm_damp_elem = elems;
        }
        if camm_onsite_scale_elem.is_empty() {
            camm_onsite_scale_elem = onsite_elem;
        }
    }
    options.camm_damp = camm_damp;
    options.camm_aes_scale = camm_aes_scale;
    options.camm_onsite_scale = camm_onsite_scale;
    options.camm_onsite_scale_elem = camm_onsite_scale_elem;
    options.camm_damp_elem = camm_damp_elem;
    options.camm_damp_charge = camm_damp_charge;
    options.lr_exchange = lr_exchange;
    options.onsite_exchange = onsite_exchange;
    options.dynamic_omega = dynamic_omega;
    options.scf_trah = scf_trah;
    if let Some(name) = multipole_secondary_basis.as_deref() {
        // Built-in name (cc-pVDZ/TZ/QZ/5Z) or a secondary-basis file path.
        let sec = if let Some(res) = crate::builtin_secondary(name) {
            res?
        } else {
            let text = std::fs::read_to_string(name).map_err(|e| {
                crate::Gfn1Error::InvalidInput(format!(
                    "--multipole-secondary-basis `{name}` is not a built-in name and cannot be \
                     read as a file: {e}"
                ))
            })?;
            crate::parse_secondary_basis(&text)?
        };
        options.multipole_secondary_basis = Some(sec);
    }
    options.charge_order = charge_order;
    if let Some(name) = scc_accel.as_deref() {
        options.scc_accelerator = match name.to_ascii_lowercase().as_str() {
            "auto" => SccAccelerator::Auto,
            "broyden" => SccAccelerator::Broyden,
            "linear" => SccAccelerator::Linear,
            "cdiis" => SccAccelerator::Cdiis,
            "newton" => SccAccelerator::Newton,
            other => {
                return Err(crate::Gfn1Error::InvalidInput(format!(
                    "unknown --scc-accel `{other}` (use auto/broyden/linear/cdiis/newton)"
                )))
            }
        };
    }

    if let Some(observable) = observable {
        let spin = parse_tda_spin(&tda_spin)?;
        return observables::run(
            observable,
            &system,
            &params,
            &options,
            observables::Settings {
                pbc: periodic_options(&options)?,
                tda: TdaOptions {
                    n_states: tda_n_states,
                    spin,
                },
                bfield: bfield.map(|[x, y, z]| Vec3::new(x, y, z)),
                e_step: field_step.unwrap_or(0.002),
                b_step: magnetic_step,
                m1_basis: m1_basis.as_deref(),
                print_matrices,
            },
        );
    }
    if print_td_dfpt {
        let soc = soc_zeta_path
            .as_ref()
            .map(|path| crate::soc::SocParameters::from_zeta_str(&std::fs::read_to_string(path)?))
            .transpose()?;
        let pbc = periodic_options(&options)?;
        let result = crate::solve_optical_td_dfpt(
            &system,
            &params,
            &options,
            &pbc,
            soc.as_ref(),
            tda_n_states,
        )?;
        println!("td_dfpt_q_fractional 0 0 0");
        println!("td_dfpt_kpoints {}", result.kpoints);
        println!("td_dfpt_transitions {}", result.transitions);
        println!(
            "td_dfpt_ground_free_energy_hartree {:.14}",
            result.ground_free_energy
        );
        println!("state excitation_Hartree excitation_eV");
        for (i, &omega) in result.excitation_energies.iter().enumerate() {
            println!(
                "{} {:.14} {:.10}",
                i + 1,
                omega,
                omega * crate::constants::HARTREE_TO_EV
            );
        }
        return Ok(());
    }
    if let Some(path) = soc_zeta_path {
        if print_third
            || print_fourth
            || optimize
            || print_vpt2
            || print_ir
            || print_raman
            || print_tda
            || print_rpa
        {
            return Err(crate::Gfn1Error::InvalidInput(
                "SOC CLI does not implement FC3, optimization, VPT2, IR/Raman or TDA".into(),
            ));
        }
        let soc = crate::soc::SocParameters::from_zeta_str(&std::fs::read_to_string(path)?)?;
        if system.lattice.is_some() {
            let pbc = periodic_options(&options)?;
            if print_polarizability {
                return Err(crate::Gfn1Error::InvalidInput(
                    "periodic SOC macroscopic electric response is not implemented".into(),
                ));
            }
            if let Some(q) = phonon_q {
                let masses: Vec<_> = system
                    .atoms
                    .iter()
                    .map(|a| crate::data_tables::relative_atomic_mass(a.z))
                    .collect();
                let reference = crate::pbc::phonon::AnalyticSupercellPhonons::build_soc(
                    &system,
                    &params,
                    &options,
                    &pbc,
                    &masses,
                    crate::pbc::phonon::PhononSupercellOptions {
                        repetitions: phonon_repetitions,
                        ..Default::default()
                    },
                    &soc,
                )?;
                let modes = reference.at_q(q)?;
                println!("phonon_backend    {}", modes.backend);
                println!("phonon_cm-1       {:?}", modes.wavenumbers_cm1);
                println!("phonon_hermiticity {:e}", modes.raw_hermiticity_error);
                println!(
                    "phonon_raw_hessian_asymmetry {:e}",
                    reference.raw_hessian_asymmetry
                );
                println!(
                    "spinor_response_residual {:?}",
                    reference.max_spinor_response_residual
                );
                if print_citations {
                    print!(
                        "{}",
                        crate::citations::bibliography(&modes.method_citations)?
                    );
                }
                return Ok(());
            }
            if phonon_repetitions != [1; 3] {
                return Err(crate::Gfn1Error::InvalidInput(
                    "--phonon-supercell requires --phonon-q".into(),
                ));
            }
            let result = if print_hessian {
                let h = crate::pbc::pbc_soc_hessian(&system, &params, &options, &pbc, &soc)?;
                println!("hessian_raw_asymmetry {:e}", h.raw_max_asymmetry);
                println!("spinor_response_residual {:e}", h.max_response_residual);
                println!("hessian_hartree_per_bohr2");
                for row in 0..h.hessian.rows() {
                    println!(
                        "{:?}",
                        (0..h.hessian.cols())
                            .map(|col| h.hessian[(row, col)])
                            .collect::<Vec<_>>()
                    );
                }
                h.soc
            } else {
                crate::pbc::run_pbc_soc_scc(&system, &params, &options, &pbc, &soc)?
            };
            println!("total_free        {:24.14}", result.total_free);
            println!("soc_energy        {:24.14}", result.soc_energy);
            println!(
                "spinor_count_per_k {:?}",
                result
                    .eigen_k
                    .iter()
                    .map(|e| e.values.len())
                    .collect::<Vec<_>>()
            );
            println!("occupation_limit  1");
            if print_gradient {
                println!("gradient");
                for g in result.analytic_gradient(&system, &params, &options, &pbc)? {
                    println!("{:24.16e} {:24.16e} {:24.16e}", g.x, g.y, g.z);
                }
            }
            if print_stress {
                println!(
                    "stress_hartree_per_bohr3 {:?}",
                    result
                        .analytic_stress(&system, &params, &options, &pbc)?
                        .as_slice()
                );
            }
            if print_charges {
                println!("charges {:?}", result.atomic_charges);
            }
            if print_citations {
                print!(
                    "{}",
                    crate::citations::bibliography(&result.method_citations)?
                );
            }
            return Ok(());
        }
        if print_hessian || print_stress || phonon_q.is_some() || phonon_repetitions != [1; 3] {
            return Err(crate::Gfn1Error::InvalidInput(
                "SOC CLI currently supports energy, charges, gradient and static polarizability"
                    .into(),
            ));
        }
        let (result, gradient) = if print_gradient {
            let (r, g) = crate::soc::soc_analytic_gradient(&system, &params, &options, &soc)?;
            (r, Some(g))
        } else {
            (
                crate::soc::run_soc_scc(&system, &params, &options, &soc)?,
                None,
            )
        };
        println!("total_free        {:24.14}", result.total_free);
        println!("soc_energy        {:24.14}", result.soc_energy);
        println!("spinor_count      {:24}", result.eigen.values.len());
        println!("occupation_limit  {:24}", 1);
        if print_polarizability {
            let a = crate::soc::soc_static_polarizability(&system, &params, &result)?;
            println!("polarizability_au {:?}", a.tensor);
        }
        if let Some(g) = gradient {
            println!("gradient");
            for x in g {
                println!("{:24.16e} {:24.16e} {:24.16e}", x.x, x.y, x.z);
            }
        }
        if print_charges {
            println!("charges {:?}", result.atomic_charges);
        }
        if print_citations {
            print!(
                "{}",
                crate::citations::bibliography(&result.method_citations)?
            );
        }
        return Ok(());
    }
    if let Some(q) = phonon_q {
        let masses: Vec<f64> = system
            .atoms
            .iter()
            .map(|a| crate::data_tables::relative_atomic_mass(a.z))
            .collect();
        let reference = crate::pbc::phonon::AnalyticSupercellPhonons::build(
            &system,
            &params,
            &options,
            &PbcOptions::for_electronic(&options)?,
            &masses,
            crate::pbc::phonon::PhononSupercellOptions {
                repetitions: phonon_repetitions,
                ..Default::default()
            },
        )?;
        let result = reference.at_q(q)?;
        println!("phonon_backend    {}", result.backend);
        println!("phonon_q_fractional {:?}", result.q_fractional);
        println!("phonon_cm-1       {:?}", result.wavenumbers_cm1);
        println!("phonon_hermiticity {:e}", result.raw_hermiticity_error);
        println!(
            "phonon_raw_hessian_asymmetry {:e}",
            reference.raw_hessian_asymmetry
        );
        if print_citations {
            print!(
                "{}",
                crate::citations::bibliography(&result.method_citations)?
            );
        }
        return Ok(());
    }
    if phonon_repetitions != [1; 3] {
        return Err(crate::Gfn1Error::InvalidInput(
            "--phonon-supercell requires --phonon-q".into(),
        ));
    }
    if print_citations {
        let mut selected = options.clone();
        selected.external_field.magnetic_field = bfield.map(|b| Vec3::new(b[0], b[1], b[2]));
        if nmr_nucleus.is_some() {
            selected.external_field.magnetic_field = Some(Vec3::zero());
        }
        print!(
            "{}",
            crate::citations::bibliography(&crate::citations::electronic_citations(
                &selected,
                system.lattice.is_some()
            ))?
        );
    }
    if let Some([bx, by, bz]) = bfield {
        options.external_field.magnetic_field = Some(Vec3::new(bx, by, bz));

        let basis = observables::load_m1_basis(m1_basis.as_deref())?;
        if print_polarizability {
            let alpha = crate::magnetic::magnetic_polarizability(
                &system,
                &params,
                &options,
                basis.as_ref(),
                field_step.unwrap_or(0.002),
            )?;
            println!("magnetic_polarizability_au {alpha:?}");
        }
        if print_gradient {
            let g = crate::magnetic::magnetic_analytic_gradient(
                &system,
                &params,
                &options,
                basis.as_ref(),
                magnetic_step.unwrap_or(0.001),
            )?;
            println!("magnetic_total {:.14}", g.energy);
            print_gradient_rows("magnetic_gradient_hartree_per_bohr", &g.gradient);
            print_gradient_rows("magnetic_forces_hartree_per_bohr", &g.forces);
            return Ok(());
        }
        let result = if let Some(basis) = &basis {
            crate::magnetic::run_magnetic_scc_m1(&system, &params, &options, basis)?
        } else {
            crate::run_magnetic_scc(&system, &params, &options)?
        };
        println!("magnetic_total    {:24.14}", result.energy);
        println!("band              {:24.14}", result.band_energy);
        println!("isotropic_scc     {:24.14}", result.scc_second_order);
        println!("third_order       {:24.14}", result.scc_third_order);
        println!("repulsion         {:24.14}", result.repulsion_energy);
        println!("dispersion        {:24.14}", result.dispersion_energy);
        println!("halogen           {:24.14}", result.halogen_energy);
        println!("iterations        {:24}", result.iterations);
        println!("converged         {:>24}", result.converged);
        return Ok(());
    }
    if let Some(nucleus) = nmr_nucleus {
        if system.lattice.is_some() {
            return Err(crate::Gfn1Error::InvalidInput(
                "NMR shielding is available for non-periodic systems only".to_string(),
            ));
        }
        if nucleus >= system.atoms.len() {
            return Err(crate::Gfn1Error::InvalidInput(format!(
                "--nmr nucleus index {nucleus} out of range ({} atoms)",
                system.atoms.len()
            )));
        }
        options.external_field.magnetic_field = Some(Vec3::zero());
        let gauge = system.atoms[nucleus].position; // common gauge origin at the nucleus
        let basis = observables::load_m1_basis(m1_basis.as_deref())?;
        let sh = crate::nmr_shielding_tensor(
            &system,
            &params,
            &options,
            basis.as_ref(),
            nucleus,
            gauge,
        )?;
        let ppm = 1.0e6;
        println!("nmr_nucleus       {:24}", nucleus);
        println!("nmr_isotropic_ppm {:24.6}", sh.isotropic() * ppm);
        println!("nmr_shielding_ppm (sigma_ab)");
        for row in &sh.sigma {
            println!(
                "{:18.6} {:18.6} {:18.6}",
                row[0] * ppm,
                row[1] * ppm,
                row[2] * ppm
            );
        }
        return Ok(());
    }

    if print_tda || print_rpa {
        let spin = parse_tda_spin(&tda_spin)?;
        let gradient = tda_grad || print_gradient;
        let hessian = excited_hessian || print_hessian;
        let periodic = system.lattice.is_some();
        let mesh = periodic_options(&options)?.kmesh;
        let mesh_tda = periodic && !mesh.is_gamma_only();
        let model = if print_rpa { "rpa" } else { "tda" };
        if periodic && (print_rpa || hessian) {
            return Err(crate::Gfn1Error::InvalidInput(
                "molecular RPA and excited-state Hessians require a molecule; use --td-dfpt for periodic full-TD optical energies".into(),
            ));
        }
        let reported_states = tda_n_states.max(tda_state.saturating_add(1));
        let td_options = TdaOptions {
            n_states: if gradient || hessian {
                usize::MAX
            } else {
                reported_states
            },
            spin,
        };
        let mut spectrum = if mesh_tda {
            crate::solve_tda_kpoint(&system, &params, &options, mesh, td_options)?
        } else if periodic {
            crate::solve_tda_pbc_gamma(&system, &params, &options, td_options)?
        } else {
            let electronic = run_electronic(&system, &params, options.clone())?;
            if print_rpa {
                crate::solve_rpa(&system, &params, &electronic, td_options)?
            } else {
                solve_tda(&system, &params, &electronic, td_options)?
            }
        };
        let td_options = TdaOptions {
            n_states: reported_states,
            spin,
        };
        println!("{model}_spin          {:>24}", spin.label());
        if periodic {
            println!("tda_kmesh {:?}", mesh.size);
        }
        let (manifold, root_separation) = if gradient || hessian {
            let manifold = spectrum.root_manifold(tda_state)?;
            let start = manifold.start;
            let end = manifold.end;
            let count = end - start;
            if count > 1 && !tda_manifold_average {
                return Err(crate::Gfn1Error::InvalidInput(format!(
                    "{model} state {tda_state} belongs to a {count}-fold degenerate manifold (0-based roots {start} through {}); individual adiabatic derivatives are undefined{}",
                    end - 1,
                    if print_rpa { "; use --rpa-manifold-average for the mean surface" } else if !mesh_tda { "; use --tda-manifold-average for the mean surface" } else { "" },
                )));
            }
            if tda_manifold_average && mesh_tda {
                return Err(crate::Gfn1Error::InvalidInput(
                    "--tda-manifold-average supports molecules and Gamma".into(),
                ));
            }
            let nearest = spectrum.root_separation(tda_state, &manifold);
            (start..end, nearest)
        } else {
            (0..0, f64::INFINITY)
        };
        spectrum.states.truncate(reported_states);

        print_tda_states(&spectrum);
        if tda_manifold_average {
            println!("{model}_manifold_first_root {}", manifold.start);
            println!("{model}_manifold_size {}", manifold.len());
            let status = if manifold.len() == 1 {
                "isolated_root"
            } else {
                "degenerate_mean"
            };
            println!("{model}_manifold_status {status}");
        }
        if root_separation.is_finite() {
            println!("{model}_root_separation_hartree {root_separation:.8e}");
        }
        if gradient {
            let nearest = root_separation;
            let g = if print_rpa {
                println!("rpa_gradient_method analytic");
                if tda_manifold_average {
                    crate::solve_rpa_manifold_gradient_analytic(
                        &system, &params, &options, tda_state, td_options,
                    )?.mean
                } else {
                    crate::solve_rpa_gradient_analytic(
                        &system, &params, &options, tda_state, td_options,
                    )?
                }
            } else {
                let method = TdaGradientMethod::parse(&tda_grad_method).ok_or_else(|| {
                    crate::Gfn1Error::InvalidInput(format!(
                        "invalid --tda-gradient-method {tda_grad_method}"
                    ))
                })?;
                let label = match method {
                    TdaGradientMethod::FiniteDifference => "finite-difference",
                    TdaGradientMethod::SemiNumerical => "semi-numerical",
                    TdaGradientMethod::Analytic => "analytic",
                };
                if method != TdaGradientMethod::Analytic
                    && !(tda_step.is_finite() && tda_step > 0.0)
                {
                    return Err(crate::Gfn1Error::InvalidInput(
                        "--tda-step must be positive and finite".into(),
                    ));
                }
                let step = crate::td::limited_tda_step(tda_step, nearest, manifold.len());
                println!("tda_gradient_method  {label}");
                if method != TdaGradientMethod::Analytic {
                    println!("tda_effective_step_bohr {step:.8e}");
                }

                if tda_manifold_average && manifold.len() > 1 {
                    crate::td::solve_tda_manifold_gradient_in_range(
                        &system, &params, &options, tda_state, manifold, td_options, step, method,
                    )?
                } else if mesh_tda {
                    match method {
                        TdaGradientMethod::Analytic => crate::solve_tda_kpoint_gradient_analytic(
                            &system, &params, &options, mesh, tda_state, td_options,
                        )?,
                        TdaGradientMethod::FiniteDifference => crate::solve_tda_kpoint_gradient(
                            &system, &params, &options, mesh, tda_state, td_options, step,
                        )?,
                        TdaGradientMethod::SemiNumerical => {
                            crate::td::solve_tda_kpoint_gradient_fd(
                                &system, &params, &options, mesh, tda_state, td_options, step,
                                false,
                            )?
                        }
                    }
                } else {
                    solve_tda_gradient_method(
                        &system, &params, &options, tda_state, td_options, step, method,
                    )?
                }
            };
            println!("{model}_state         {:24}", tda_state + 1);
            println!("{model}_total_energy  {:24.14}", g.total_energy);

            let label = if tda_manifold_average {
                format!("{model}_manifold_average_gradient (Hartree/bohr)")
            } else {
                format!("{model}_gradient (state {}, Hartree/bohr)", tda_state + 1)
            };
            print_gradient_rows(&label, &g.gradient);
        }
        if hessian {
            let h = if print_rpa {
                if tda_manifold_average {
                    crate::solve_rpa_manifold_hessian_analytic(
                        &system, &params, &options, tda_state, td_options,
                    )?.mean
                } else {
                    crate::solve_rpa_hessian_analytic(
                        &system, &params, &options, tda_state, td_options,
                    )?
                }
            } else if tda_manifold_average {
                crate::solve_tda_manifold_hessian_analytic(
                    &system, &params, &options, tda_state, td_options,
                )?
                .mean
            } else {
                crate::solve_tda_hessian_analytic(
                    &system, &params, &options, tda_state, td_options,
                )?
            };
            println!("{model}_total_energy  {:24.14}", h.total_energy);

            let label = if tda_manifold_average {
                format!("{model}_manifold_average_hessian_hartree_per_bohr2")
            } else {
                format!("{model}_hessian_hartree_per_bohr2")
            };
            print_matrix(&label, &h.hessian);
        }
        return Ok(());
    }
    if print_polarizability || print_ir || print_raman {
        if system.lattice.is_some() {
            return Err(crate::Gfn1Error::InvalidInput(
                "IR / Raman / polarizability are available for non-periodic systems only"
                    .to_string(),
            ));
        }
        if print_polarizability {
            let electronic = run_electronic(&system, &params, options.clone())?;
            let pol = static_polarizability(&system, &params, &electronic)?;
            println!(
                "dipole_au         {:16.10} {:16.10} {:16.10}",
                electronic.dipole.x, electronic.dipole.y, electronic.dipole.z
            );
            println!("polarizability_au (alpha_ij)");
            for row in pol.tensor {
                println!("{:18.10} {:18.10} {:18.10}", row[0], row[1], row[2]);
            }
            println!("polar_isotropic   {:24.10}", pol.isotropic);
            println!("polar_anisotropy  {:24.10}", pol.anisotropy);
        }
        if print_ir {
            let hess_options = AnalyticHessianOptions {
                electronic_options: options.clone(),
                ..AnalyticHessianOptions::default()
            };
            let ir = ir_spectrum(&system, &params, hess_options, Vec3::zero())?;
            println!("ir_spectrum (wavenumber_cm-1  intensity_km/mol  intensity_au)");
            for m in &ir.modes {
                println!(
                    "{:14.4} {:18.8} {:18.10}",
                    m.wavenumber, m.intensity_km_per_mol, m.intensity_au
                );
            }
            print_dipole_derivatives(&ir.dipole_derivatives.ddipole_dr);
        }
        if print_raman {
            let hess_options = AnalyticHessianOptions {
                electronic_options: options.clone(),
                ..AnalyticHessianOptions::default()
            };
            let raman = raman_spectrum(
                &system,
                &params,
                hess_options,
                Vec3::zero(),
                field_step.unwrap_or(1e-3),
            )?;
            println!("raman_spectrum (wavenumber_cm-1  activity_au  mean_a'  gamma'^2)");
            for m in &raman.modes {
                println!(
                    "{:14.4} {:18.8} {:16.8} {:16.8}",
                    m.wavenumber,
                    m.activity,
                    m.mean_polarizability_derivative,
                    m.anisotropy_squared
                );
            }
            print_polarizability_derivatives(&raman.polarizability_derivatives.dpolarizability_dr);
        }
        return Ok(());
    }
    if param_deriv {
        let mut targets: Vec<ParameterTarget> = if all_param_targets {
            active_targets_for_system(&params, &system)
        } else {
            let spec = param_targets.ok_or_else(|| {
                crate::Gfn1Error::InvalidInput(
                    "--param-deriv needs --targets a,b,... or --all-param-targets".to_string(),
                )
            })?;
            spec.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(ParameterTarget::parse)
                .collect::<Result<Vec<_>>>()?
        };
        let total_targets = targets.len();
        // Optional `i/n` chunk selection to restrict (suppress) the output to a
        // contiguous slice of targets.
        if let Some(spec) = target_chunk.as_deref() {
            let (idx, count) = spec.split_once('/').ok_or_else(|| {
                crate::Gfn1Error::InvalidInput("--target-chunk must be `i/n` (1-based)".to_string())
            })?;
            let idx = idx.trim().parse::<usize>().map_err(|_| {
                crate::Gfn1Error::InvalidInput(format!("invalid chunk index `{idx}`"))
            })?;
            let count = count.trim().parse::<usize>().map_err(|_| {
                crate::Gfn1Error::InvalidInput(format!("invalid chunk count `{count}`"))
            })?;
            targets = crate::select_target_chunk(targets, idx, count)?;
        }
        let pd_options = ParamDerivativeOptions {
            step: param_step,
            electronic: options.clone(),
            include_forces: param_forces,
            include_stress: param_stress,
        };
        let derivs = parameter_finite_difference(&system, &params, &targets, &pd_options)?;
        // Tab-separated output: one block per target with the requested observables.
        println!("# param_step\t{param_step:.6e}");
        println!("# targets_total\t{total_targets}");
        println!("# targets_evaluated\t{}", derivs.len());
        println!("target\tvalue\tdE/dp");
        for d in &derivs {
            println!(
                "{}\t{:.12e}\t{:.14e}",
                d.target.label(),
                d.value,
                d.energy_derivative
            );
            if let Some(force_derivs) = &d.force_derivatives {
                for (iat, f) in force_derivs.iter().enumerate() {
                    println!(
                        "  dF/dp\tatom_{}\t{:.14e}\t{:.14e}\t{:.14e}",
                        iat + 1,
                        f.x,
                        f.y,
                        f.z
                    );
                }
            }
            if let Some(s) = &d.stress_derivative {
                println!(
                    "  dStress/dp\t{:.10e}\t{:.10e}\t{:.10e}\t{:.10e}\t{:.10e}\t{:.10e}",
                    s[0][0], s[1][1], s[2][2], s[1][2], s[0][2], s[0][1]
                );
            }
        }
        return Ok(());
    }
    if optimize {
        let mut grad_options = AnalyticGradientOptions::default();
        grad_options.electronic = options.clone();
        let mut opt_options = GeometryOptimizationOptions {
            gradient_options: grad_options,
            ..GeometryOptimizationOptions::default()
        };
        if print_vpt2 {
            opt_options.gradient_tolerance = 1e-6;
        }
        if let Some(max_iter) = opt_max_iter {
            opt_options.max_iterations = max_iter;
        }
        if let Some(gtol) = opt_gtol {
            opt_options.gradient_tolerance = gtol;
        }
        // Stream the XYZ trajectory live (one flushed frame per step) during optimization.
        opt_options.trajectory_path = opt_traj.as_ref().map(std::path::PathBuf::from);
        let result = optimize_geometry(&system, &params, opt_options)?;
        println!("opt_converged     {:>24}", result.converged);
        println!("opt_iterations    {:24}", result.iterations);
        println!("opt_energy        {:24.14}", result.energy);
        println!("opt_max_gradient  {:24.14}", result.max_gradient);
        if let Some(path) = &opt_traj {
            println!("opt_trajectory    {:>24}", path);
        }
        // Always persist the optimized geometry to a file (it would otherwise be lost on exit).
        // `--opt-output PATH` overrides; the default is `<input-stem>_opt.xyz` beside the input.
        let out_path = opt_output.unwrap_or_else(|| {
            let p = std::path::Path::new(&xyz);
            let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("geometry");
            let parent = p.parent().filter(|d| !d.as_os_str().is_empty());
            let name = format!("{stem}_opt.xyz");
            match parent {
                Some(dir) => dir.join(name).to_string_lossy().into_owned(),
                None => name,
            }
        });
        std::fs::write(&out_path, xyz_string(&result.system))?;
        println!("opt_output        {out_path:>24}");
        if print_vpt2 {
            print_vpt2_spectrum(&result.system, &params, options)?;
        }
        return Ok(());
    }
    if print_vpt2 {
        return print_vpt2_spectrum(&system, &params, options);
    }

    if print_gradient && !dfpt && kmesh.is_some() {
        let pbc = periodic_options(&options)?;
        let result = crate::pbc_analytic_gradient(&system, &params, &options, &pbc)?;
        print_energy_terms(&crate::pbc_electronic_result(
            result.scf,
            &system,
            pbc.ao_cutoff,
        )?);
        print_gradient_rows("gradient", &result.gradient);
        print_gradient_rows("forces", &result.forces);
        return Ok(());
    }
    if print_gradient && !dfpt {
        let mut grad_options = AnalyticGradientOptions::default();
        grad_options.electronic = options.clone();
        let result = analytic_gradient(&system, &params, grad_options)?;
        print_energy_terms(&result.electronic_result);
        println!("gradient");
        for g in &result.gradient {
            println!("{:24.16e} {:24.16e} {:24.16e}", g.x, g.y, g.z);
        }
        println!("forces");
        for f in &result.forces {
            println!("{:24.16e} {:24.16e} {:24.16e}", f.x, f.y, f.z);
        }
        return Ok(());
    }
    if print_hessian {
        if system.lattice.is_some() {
            let pbc = periodic_options(&options)?;
            let result = if pbc.kmesh.is_gamma_only() {
                crate::pbc_gamma_hessian(&system, &params, &options, &pbc)?
            } else {
                crate::pbc_kpoint_hessian(&system, &params, &options, &pbc)?
            };

            if dfpt && print_gradient {
                let gradient = crate::pbc::pbc_gradient_from_scc(
                    &system,
                    &params,
                    result.scf.clone(),
                    &options,
                    &pbc,
                )?;
                print_gradient_rows("gradient", &gradient.gradient);
                print_gradient_rows("forces", &gradient.forces);
            }
            print_energy_terms(&crate::pbc_electronic_result(
                result.scf,
                &system,
                pbc.ao_cutoff,
            )?);
            println!("hessian_size      {:24}", result.hessian.rows());
            println!("hessian_raw_asymmetry {:e}", result.raw_max_asymmetry);
            if print_matrices || dfpt {
                print_matrix("hessian", &result.hessian);
            }
            return Ok(());
        }
        let hess_options = AnalyticHessianOptions {
            electronic_options: options.clone(),
            ..AnalyticHessianOptions::default()
        };
        let result = analytic_hessian(&system, &params, hess_options)?;
        if let Some(electronic) = &result.electronic_result {
            print_energy_terms(electronic);
        }
        println!("hessian_size      {:24}", result.hessian.rows());
        if print_matrices {
            print_matrix("hessian", &result.hessian);
        }
        return Ok(());
    }
    if print_third || print_fourth {
        options.energy_tolerance = options.energy_tolerance.min(1e-12);
        options.charge_tolerance = options.charge_tolerance.min(1e-11);
    }
    if print_fourth {
        if quartic_file.is_none() && system.atoms.len() * 3 > crate::MAX_FOURTH_DERIVATIVE_NDOF {
            return Err(crate::Gfn1Error::InvalidInput(format!(
                "quartic text output is capped at {} nuclear coordinates; use --quartic-file PATH for a packed result",
                crate::MAX_FOURTH_DERIVATIVE_NDOF
            )));
        }
        if system.lattice.is_some() {
            return Err(crate::Gfn1Error::InvalidInput(
                "analytic periodic FC4 is unavailable".into(),
            ));
        }
        let h = AnalyticHessianOptions {
            electronic_options: options.clone(),
            ..Default::default()
        };
        let cutoff = options.hamiltonian.coordination_cutoff;
        if let Some(path) = quartic_file {
            let storage = crate::DerivativeStorageOptions {
                disk_storage: true, output_path: Some(path), ..Default::default()
            };
            let crate::DerivativeOutput::File(file) =
                crate::fourth_derivative_with_storage(&system, &params, &h, cutoff, &storage)?
            else { unreachable!("disk storage requested") };
            print_derivative_file("quartic", &file);
        } else {
            let q = crate::fourth_derivative_analytic_dense(&system, &params, &h, cutoff)?;
            print_quartic(&q)?;
        }
    }
    if print_third {
        if let Some(path) = cubic_file {
            let h = AnalyticHessianOptions {
                electronic_options: options.clone(), ..Default::default()
            };
            let cutoff = options.hamiltonian.coordination_cutoff;
            let pbc = if system.lattice.is_some() { Some(periodic_options(&options)?) } else { None };
            let storage = crate::DerivativeStorageOptions {
                disk_storage: true, output_path: Some(path), ..Default::default()
            };
            let crate::DerivativeOutput::File(file) =
                crate::third_derivative_with_storage(&system, &params, &h, cutoff, pbc.as_ref(), &storage)?
            else { unreachable!("disk storage requested") };
            print_derivative_file("cubic", &file);
            return Ok(());
        }
        if system.lattice.is_some() {
            let pbc = periodic_options(&options)?;
            let tensor = if pbc.kmesh.is_gamma_only() {
                crate::pbc_gamma_third_analytic_dense(&system, &params, &options, &pbc)?
            } else {
                crate::pbc_kpoint_third_analytic_dense(&system, &params, &options, &pbc)?
            };
            print_cubic(&tensor)?;
            return Ok(());
        }
        let hess_options = AnalyticHessianOptions {
            electronic_options: options.clone(),
            ..AnalyticHessianOptions::default()
        };
        let cutoff = options.hamiltonian.coordination_cutoff;
        let tensor = crate::third_derivative_analytic_dense(&system, &params, hess_options, cutoff)?;
        print_cubic(&tensor)?;
    }
    if print_third || print_fourth {
        return Ok(());
    }
    if print_stress {
        if system.lattice.is_none() {
            return Err(crate::Gfn1Error::InvalidInput(
                "stress requires a periodic input with Lattice/PBC".to_string(),
            ));
        }
        let pbc = periodic_options(&options)?;
        let result = pbc_stress(&system, &params, &options, &pbc)?;
        print_energy_terms(&crate::pbc_electronic_result(
            result.scf.clone(),
            &system,
            pbc.ao_cutoff,
        )?);
        print_matrix("stress", &result.stress);
        return Ok(());
    }

    let result = if kmesh.is_some() {
        let pbc = periodic_options(&options)?;
        crate::pbc_electronic_result(
            crate::run_pbc_scc(&system, &params, &options, &pbc)?,
            &system,
            pbc.ao_cutoff,
        )?
    } else {
        run_electronic(&system, &params, options)?
    };
    print_energy_terms(&result);
    if print_charges {
        println!("atomic_charges");
        for (iat, charge) in result.atomic_charges.iter().enumerate() {
            println!("{:5} {:24.14}", iat + 1, charge);
        }
        println!("shell_charges");
        for (ish, charge) in result.shell_charges.iter().enumerate() {
            println!("{:5} {:24.14}", ish + 1, charge);
        }
    }
    if print_matrices {
        print_vector("orbital_energies", &result.orbital_energies);
        print_matrix("overlap", &result.integrals.overlap);
        print_matrix("h0", &result.h0);
        print_matrix("fock", &result.fock);
        print_matrix("density", &result.density);
    }
    Ok(())
}

fn parse_tda_spin(value: &str) -> Result<TdaSpin> {
    match value.to_ascii_lowercase().as_str() {
        "singlet" | "s" => Ok(TdaSpin::Singlet),
        "triplet" | "t" => Ok(TdaSpin::Triplet),
        _ => Err(crate::Gfn1Error::InvalidInput(format!(
            "unknown TD-GFN1 spin {value}; use singlet or triplet"
        ))),
    }
}

fn parse_number(args: &mut impl Iterator<Item = String>, name: &str) -> Result<f64> {
    let value = required(args, name)?;
    let number = value
        .parse::<f64>()
        .map_err(|_| crate::Gfn1Error::InvalidInput(format!("invalid {name} value {value}")))?;
    if !number.is_finite() {
        return Err(crate::Gfn1Error::InvalidInput(format!(
            "{name} requires a finite number"
        )));
    }
    Ok(number)
}

fn required(args: &mut impl Iterator<Item = String>, name: &str) -> Result<String> {
    args.next()
        .ok_or_else(|| crate::Gfn1Error::InvalidInput(format!("{name} requires a value")))
}

fn print_gradient_rows(label: &str, gradient: &[Vec3]) {
    println!("{label}");
    for g in gradient {
        println!("{:24.14e} {:24.14e} {:24.14e}", g.x, g.y, g.z);
    }
}

fn parse_three<T: std::str::FromStr>(value: String, name: &str) -> Result<[T; 3]> {
    let values = value
        .split(',')
        .map(|x| {
            x.parse().map_err(|_| {
                crate::Gfn1Error::InvalidInput(format!("invalid {name} value `{value}`"))
            })
        })
        .collect::<Result<Vec<T>>>()?;
    values.try_into().map_err(|_| {
        crate::Gfn1Error::InvalidInput(format!(
            "{name} requires exactly three comma-separated values"
        ))
    })
}

fn print_vpt2_spectrum(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: ElectronicOptions,
) -> Result<()> {
    let masses: Vec<f64> = system
        .atoms
        .iter()
        .map(|a| crate::data_tables::relative_atomic_mass(a.z))
        .collect();
    let result =
        crate::molecular_vpt2_pes(system, params, options, &masses, Default::default(), 1e-5)?;
    println!("vpt2_scope        {}", result.spectrum.scope);
    println!(
        "vpt2_zpe_hartree  {:24.14}",
        result.spectrum.ground.energy_hartree()
    );
    println!("mode harmonic_cm-1 fundamental_cm-1");
    for (i, (a, b)) in result
        .normal_modes
        .vibrations
        .wavenumbers
        .iter()
        .zip(&result.spectrum.fundamentals_cm1)
        .enumerate()
    {
        println!("{} {:.6} {:.6}", i + 1, a, b);
    }
    Ok(())
}

fn print_help() {
    println!("  properties --request FILE [--output FILE]  Structured crystal-property workflow; see properties --help");
    println!("  --log-file PATH --log-level error|info|debug|trace  Append calculation events as JSON Lines");
    println!("  --threads N --memory-mb N  Worker count and checked workspace budget (MiB)");
    println!(
        "  --cp-tolerance X --cp-relative-tolerance X --cp-max-iter N  CP/DFPT residual controls"
    );
    println!("  --phonon-q qx,qy,qz --phonon-supercell nx,ny,nz  Analytic commensurate phonons");
    println!("  --kmesh nx,ny,nz [--gamma-centered]  Explicit periodic electronic mesh");
    println!("  --dfpt  Analytic periodic Hessian with Cartesian components");
    println!(
        "  --soc, --soc-zeta-file PATH  Explicit zeta L.S parameters (hartree) with provenance"
    );
    println!("  --vpt2-pes           PES-only analytic VPT2 at a stationary minimum");
    println!(
        "  --ml-model, --ml-xtb PATH|MODEL  Embedded or user-supplied radial Hamiltonian weights"
    );
    println!("  --list-ml-models     List embedded models including builtin:water-clusters");
    println!("  --alpb SOLVENT | --gbsa SOLVENT   GFN1-specific equilibrium solvation");
    println!("  --solvent-state gsolv|bar1mol|reference --solvent-temperature K --solvent-grid N");
    println!("  --citations       Print selected method references as BibTeX (XYZ optional)");
    println!("  --pbc-ao-cutoff B  Periodic AO image-sum cutoff in bohr (default 30)");
    println!(
        r#"Usage:
  gfn1-rs [OPTIONS] molecule.xyz

Input and electronic state:
  --param FILE                    GFN1-xTB parameter file. Also accepts the
                                  builtin specs `builtin` (GFN1-xTB) and
                                  `builtin:si` (GFN1(Si)-xTB). Resolution:
                                  --param > GFN1_XTB_PARAM > builtin.
  --charge Q                      Total molecular charge.
  --multiplicity M, --spin-multiplicity M
                                  Spin multiplicity for occupations.
  --spinpol, --spin-polarization  Spin-polarized GFN1 ("spGFN1"): add the W spin
                                  term + spin-unrestricted SCC for open shells
                                  (closed shells stay byte-identical to GFN1).
  --plus-u                        DFT+U on the correlated (transition-metal d)
                                  shell (requires --spinpol; open-shell only).
  --hubbard-u Elem:U,...          Fixed on-site U (Hartree) per element, e.g.
                                  `Fe:0.15`. Implies --plus-u.
  --plus-u-v / --hubbard-v E:E:V  Add the inter-site +V term (metal–ligand
                                  hybridisation), e.g. `--hubbard-v Fe:N:0.04`.
  --hubbard-v-cutoff R            +V neighbour cutoff in bohr (default 10).
  --linear-response-u             Compute U (and V) NON-EMPIRICALLY by linear
                                  response — no fitted parameters. Implies
                                  --plus-u; auto-selects the TM d shells.
  --plus-u-all-d                  With --linear-response-u, apply +U to ALL atoms
                                  with a d shell (incl. main-group d polarisation),
                                  not just transition metals. Implies --plus-u.
  --bohr                          Read XYZ coordinates in Bohr instead of Angstrom.
  --max-scc N                     Maximum SCC iterations.
  --electronic-temperature K, --etemp K
                                  Fermi electronic temperature in Kelvin.

Dispersion:
  --no-dispersion                 Disable dispersion entirely.
  --d3-reference FILE             Override bundled simple-dftd3 reference data.
  --experimental-d4, --d4         Use experimental self-consistent D4
                                  dispersion instead of D3; non-PBC only.
                                  D4 a1/a2/s8 are read from --param FILE;
                                  s9 defaults to the GFN2 value 5.0.
  --d4-cutoff R                   D4 pair cutoff in Bohr (default 60).
  --d4-cn-cutoff R                D4 coordination-number cutoff in Bohr (default 30).
  --d4-s9 S                       D4 ATM scale factor (default 5.0; 0 disables).
  --no-d4-atm                     Disable the D4 ATM / three-body term.
  --d4-atm-cutoff R               D4 ATM cutoff in Bohr (default 40).

SCC controls:
  --no-broyden                    Use linear charge mixing.
  --mixing X                      Linear/Broyden damping factor.
  --broyden-size N                Broyden history size.
  --scc-accel auto|broyden|linear|cdiis|newton
                                  Select SCC accelerator.
  --level-shift B                 Virtual level shift in Hartree.
  --charge-order N                Experimental on-site charge expansion order
                                  (3 = stock GFN1).

Experimental electrostatics and exchange:
  --multipole                     Enable mDFTB2 multipole electrostatics.
  --multipole-octupole            Add octupole terms to the legacy multipole path.
  --field-multipole               Couple mDFTB2 dipoles to --field.
  --multipole-third-order         Add third-order on-site multipole cross terms.
  --multipole-secondary-basis NAME|FILE
                                  Use a richer secondary basis for multipoles.
  --multipole-order N             Highest arbitrary multipole rank; >=4 selects
                                  the generic rank path.
  --multipole-charge-order LIST   Per-rank charge cross-term orders, e.g. 6,4,2.
  --lr-exchange                   Enable experimental long-range Fock exchange.
  --onsite-exchange               Add exact one-center exchange correction
                                  (implies --lr-exchange).
  --dynamic-omega                 Geometry-adaptive range separation
                                  (implies --lr-exchange).
  --scf-trah                      Use TRAH second-order SCF for exchange runs.

Outputs:
  total_internal is the energy E at the converged electronic occupations.
  total_free is E - T_elec*S_elec. The entropy line is -T_elec*S_elec.
  These values are reported in Hartree.
  --gradient, --grad              Print analytic nuclear gradient.
  --hessian, --hess               Print analytic Hessian.
  --third-derivative, --cubic     Print analytic third derivative.
  --fourth-derivative, --quartic  Analytic quartic tensor (molecular; finite T supported).
  --cubic-file PATH              Stream packed FC3 to a binary file.
  --quartic-file PATH            Stream packed FC4 to a binary file without the text DOF cap.
  --stress                        Print periodic stress when the input has a cell.
  --charges                       Print Mulliken charges.
  --matrices                      Print selected internal matrices.

Optimization:
  --optimize, --opt               Run native L-BFGS geometry optimization.
  --opt-max-iter N                Maximum optimization iterations.
  --opt-gtol G                    Optimization gradient tolerance.
  --opt-output FILE               Write final optimized XYZ.
  --opt-traj FILE                 Write optimization trajectory XYZ.

Fields and properties:
  --field Ex Ey Ez                Uniform electric field in atomic units.
  --bfield Bx By Bz               Uniform magnetic field in atomic units.
  --polarizability, --polar       Static dipole polarizability.
  --ir                            IR intensities.
  --raman                         Raman activities.
  --field-step H                  Finite-field step for field properties.

  --nmr ATOM                      NMR shielding for 0-based atom index.
  --magnetizability               Static magnetic susceptibility tensor.
  --magnetic-polarizability       Electric polarizability at --bfield (default zero).
  --cotton-mouton                 Static d² alpha_ij / dB_k² at zero magnetic field.
  --mcd                           Static d alpha_ij / dB_k at --bfield (default zero).
  --magnetic-field-step H         Magnetic field step in atomic units.
  --m1-basis NAME|FILE            Secondary basis for the molecular magnetic model.
  --ecd | --optical-rotation E    Molecular rotatory strengths or beta at E Hartree.
  --piezoelectric | --born-charges Proper clamped-ion piezoelectricity and Born charges.
  --dielectric | --berry-polarization  Electronic dielectric or Berry polarization.
  --electromechanical             Coupled clamped and relaxed crystal response.
  --gruneisen | --gruneisen-second-order  Periodic mode Gruneisen parameters.
  --strain-hessian-derivative H   Periodic dH/d(volume strain), with strain step H.
  --frequencies                   Harmonic frequencies in cm^-1.
  --fmo-fragments FILE | --dc-fragments FILE   Fragment calculation with JSON partition.

Excited states:
  --rpa, --full-td                Molecular full TD-GFN1 spectrum.
  --rpa-grad, --rpa-hessian       Analytic molecular RPA gradient or Hessian.
  --rpa-manifold-average          Mean gradient or Hessian of the selected RPA manifold.
  --td-dfpt                       Optical periodic full-TD spectrum, with optional --soc.
  --tda                           Solve TDA excited states.
  --stda-xtb                      Native Rust sTDA-xTB with its dedicated orbitals.
  --stda-window-ev E               sTDA excitation selection window (default 10 eV).
  --stda-triplet                   Select triplet sTDA-xTB excitations.
  --n-states N                    Number of TDA states.
  --tda-spin singlet|triplet      TDA spin block.
  --tda-grad                      Print excited-state gradient.

  --tda-hessian                   Analytic molecular TDA Hessian.
  --tda-manifold-average          Mean gradient or Hessian of the selected TDA manifold.
  --stda-grad, --stda-hessian     Analytic derivatives of GFN1 free energy plus sTDA excitation.
  --stda-manifold-average         Mean gradient or Hessian of the selected ensemble sTDA manifold.
  --stda-thermal / --stda-legacy  Explicitly select the ensemble or fixed-partition sTDA model.
  --tda-state K                   0-based TDA state for a gradient or Hessian.
  --tda-step H                    Maximum TDA displacement in bohr. Reduced near close roots.
  --tda-gradient-method METHOD    analytic (default)|semi-numerical|finite-difference.

Parameter derivatives:
  --param-deriv                   Print finite-difference parameter derivatives.
  --targets SPEC                  Target list: glob:ks, elem:1:GAM, pair:1:6, ...
  --all-param-targets             Use all active parameter targets.
  --param-forces                  Include force derivatives.
  --param-stress                  Include stress derivatives.
  --target-chunk I/N              Evaluate only chunk I of N target chunks.
  --param-step H                  Parameter finite-difference step.

Other:
  -V, --version                   Show the package version.
  -h, --help                      Show this help.
"#
    );
}

fn print_energy_terms(result: &crate::ElectronicResult) {
    println!("total_free        {:24.14}", result.total_free);
    if result.solvation.is_some() {
        println!(
            "solv_electrostatic {:24.14}",
            result.solvation_energy.electrostatic
        );
        println!(
            "solv_hbond         {:24.14}",
            result.solvation_energy.hydrogen_bond
        );
        println!("solv_sasa          {:24.14}", result.solvation_energy.sasa);
        println!("solv_shift         {:24.14}", result.solvation_energy.shift);
    }
    println!("total_internal    {:24.14}", result.total_internal);
    println!("electronic        {:24.14}", result.electronic_energy);
    println!("repulsion         {:24.14}", result.repulsion_energy);
    println!("isotropic_scc     {:24.14}", result.isotropic_scc_energy);
    println!("third_order       {:24.14}", result.third_order_energy);
    println!("dispersion        {:24.14}", result.dispersion_energy);
    println!("halogen           {:24.14}", result.halogen_energy);
    if result.external_field_energy != 0.0 {
        println!("external_field    {:24.14}", result.external_field_energy);
    }
    println!("entropy           {:24.14}", result.electronic_entropy_term);
    println!(
        "dipole_au         {:16.10} {:16.10} {:16.10}",
        result.dipole.x, result.dipole.y, result.dipole.z
    );
    println!("iterations        {:24}", result.iterations);
    println!("converged         {:>24}", result.converged);
}

fn xyz_string(system: &PeriodicSystem) -> String {
    let bohr_to_angstrom = 1.0 / crate::system::ANGSTROM_TO_BOHR;
    // Preserve periodicity: a periodic optimized geometry is written as extended XYZ with a
    // `Lattice="..." pbc="..."` comment so it round-trips through `from_xyz_str` (the cell would
    // otherwise be silently dropped, turning a relaxed crystal into a bare molecule).
    let comment = match &system.lattice {
        Some(lat) => {
            let v = |k: usize| lat.cell.column(k) * bohr_to_angstrom;
            let (a, b, c) = (v(0), v(1), v(2));
            let pbc = |flag: bool| if flag { "T" } else { "F" };
            format!(
                "Lattice=\"{:.10} {:.10} {:.10} {:.10} {:.10} {:.10} {:.10} {:.10} {:.10}\" \
                 pbc=\"{} {} {}\" gfn1-rs optimized geometry",
                a.x,
                a.y,
                a.z,
                b.x,
                b.y,
                b.z,
                c.x,
                c.y,
                c.z,
                pbc(lat.periodic[0]),
                pbc(lat.periodic[1]),
                pbc(lat.periodic[2])
            )
        }
        None => "gfn1-rs optimized geometry".to_string(),
    };
    let mut out = format!("{}\n{comment}\n", system.atoms.len());
    for atom in &system.atoms {
        let symbol = crate::z_to_symbol(atom.z).unwrap_or("X");
        out.push_str(&format!(
            "{symbol:2} {:18.10} {:18.10} {:18.10}\n",
            atom.position.x * bohr_to_angstrom,
            atom.position.y * bohr_to_angstrom,
            atom.position.z * bohr_to_angstrom
        ));
    }
    out
}

fn print_vector(name: &str, values: &[f64]) {
    println!("{name}");
    for value in values {
        print!(" {:24.16e}", value);
    }
    println!();
}

fn print_matrix(name: &str, matrix: &crate::linalg::Matrix) {
    println!("{name} {} {}", matrix.rows(), matrix.cols());
    for i in 0..matrix.rows() {
        for j in 0..matrix.cols() {
            print!(" {:24.16e}", matrix[(i, j)]);
        }
        println!();
    }
}

fn print_derivative_file(label: &str, file: &crate::PackedDerivativeMetadata) {
    println!("{label}_file {}", file.path.display());
    println!("{label}_dof {}", file.n_dof);
    println!("{label}_packed_entries {}", file.packed_entries);
    println!("{label}_units {}", file.units());
}

fn print_cubic(tensor: &crate::SymmetricThird) -> Result<()> {
    use std::io::Write;
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    let n = tensor.n();
    writeln!(out, "third_derivative_size {n:24}")?;
    writeln!(out, "cubic_indices_0_based units=Hartree/bohr^3; slab c, row a, column b")?;
    for c in 0..n {
        writeln!(out, "third_derivative_slab[c={c}] {n} {n}")?;
        for a in 0..n {
            for b in 0..n {
                write!(out, " {:24.16e}", tensor.get(a, b, c))?;
            }
            writeln!(out)?;
        }
    }
    Ok(out.flush()?)
}

fn print_quartic(tensor: &crate::fourth_derivative::SymmetricFourth) -> Result<()> {
    use std::io::Write;
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    let n = tensor.n();
    writeln!(out, "fourth_derivative_size {n:24}")?;
    writeln!(out, "quartic_indices_0_based units=Hartree/bohr^4")?;
    for a in 0..n {
        for b in 0..n {
            for c in 0..n {
                for d in 0..n {
                    writeln!(out, "quartic {a} {b} {c} {d} {:.14e}", tensor.get(a, b, c, d))?;
                }
            }
        }
    }
    Ok(out.flush()?)
}

fn print_tda_states(result: &crate::TdaResult) {
    println!(
        "{:>5} {:>18} {:>18} {:>18}",
        "state", "energy_eV", "energy_Hartree", "osc_strength"
    );
    for (idx, st) in result.states.iter().enumerate() {
        println!(
            "{:>5} {:>18.6} {:>18.10} {:>18.8}",
            idx + 1,
            st.excitation_energy * crate::constants::HARTREE_TO_EV,
            st.excitation_energy,
            st.oscillator_strength
        );
    }
}

fn print_dipole_derivatives(ddip: &[[f64; 3]]) {
    println!("dipole_derivatives dmu/dR  (coord=3*atom+axis: dmu_x dmu_y dmu_z, atomic units)");
    for (coord, row) in ddip.iter().enumerate() {
        println!(
            "{coord:5} {:22.14e} {:22.14e} {:22.14e}",
            row[0], row[1], row[2]
        );
    }
}

fn print_polarizability_derivatives(dpolar: &[[[f64; 3]; 3]]) {
    println!(
        "polarizability_derivatives dalpha/dR  (coord: a_xx a_yy a_zz a_xy a_xz a_yz, atomic units)"
    );
    for (coord, t) in dpolar.iter().enumerate() {
        println!(
            "{coord:5} {:18.10e} {:18.10e} {:18.10e} {:18.10e} {:18.10e} {:18.10e}",
            t[0][0], t[1][1], t[2][2], t[0][1], t[0][2], t[1][2]
        );
    }
}

fn print_stda_response_metadata(thermal: bool, roots: &std::ops::Range<usize>, average: bool) {
    let model = if thermal { "thermal_ensemble" } else { "legacy_fixed_partition" };
    println!("stda_response_model {model}");
    if average {
        println!("stda_manifold_first_root {}", roots.start);
        println!("stda_manifold_size {}", roots.len());
        let status = if roots.len() == 1 { "isolated_root" } else { "degenerate_mean" };
        println!("stda_manifold_status {status}");
    }
}
