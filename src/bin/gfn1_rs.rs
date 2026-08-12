// SPDX-License-Identifier: GPL-3.0-or-later

use gfn1_rs::math::Vec3;
use gfn1_rs::{
    active_targets_for_system, analytic_gradient, analytic_hessian, ir_spectrum, optimize_geometry,
    parameter_finite_difference, pbc_stress, raman_spectrum, run_electronic,
    solve_tda, solve_tda_gradient_method, static_polarizability, AnalyticGradientOptions,
    AnalyticHessianOptions, ElectronicOptions, GeometryOptimizationOptions, Gfn1Parameters,
    ParamDerivativeOptions, ParameterTarget, PbcOptions, PeriodicSystem, Result, SccAccelerator,
    TdaGradientMethod, TdaOptions, TdaSpin,
};

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
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
    let mut field_step = 1.0e-3;
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
    let mut tda_n_states = 5usize;
    let mut tda_spin = "singlet".to_string();
    let mut tda_grad = false;
    let mut tda_state = 0usize;
    let mut tda_step = 1.0e-3;
    let mut tda_grad_method = "semi-numerical".to_string();
    let mut nmr_nucleus: Option<usize> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--param" => param = args.next(),
            "--d3-reference" => d3_reference = args.next(),
            "--experimental-d4" | "--d4" => experimental_d4 = true,
            "--d4-cutoff" => {
                let value = args.next().ok_or_else(|| {
                    gfn1_rs::Gfn1Error::InvalidInput("--d4-cutoff needs a value".to_string())
                })?;
                d4_cutoff = Some(value.parse::<f64>().map_err(|_| {
                    gfn1_rs::Gfn1Error::InvalidInput(format!("invalid --d4-cutoff `{value}`"))
                })?);
            }
            "--d4-cn-cutoff" => {
                let value = args.next().ok_or_else(|| {
                    gfn1_rs::Gfn1Error::InvalidInput("--d4-cn-cutoff needs a value".to_string())
                })?;
                d4_cn_cutoff = Some(value.parse::<f64>().map_err(|_| {
                    gfn1_rs::Gfn1Error::InvalidInput(format!("invalid --d4-cn-cutoff `{value}`"))
                })?);
            }
            "--no-d4-atm" | "--no-d4-threebody" | "--no-d4-three-body" | "--no-d4-3body" => {
                d4_atm = false;
            }
            "--d4-atm-cutoff" | "--d4-threebody-cutoff" | "--d4-3body-cutoff" => {
                let value = args.next().ok_or_else(|| {
                    gfn1_rs::Gfn1Error::InvalidInput("--d4-atm-cutoff needs a value".to_string())
                })?;
                d4_atm_cutoff = Some(value.parse::<f64>().map_err(|_| {
                    gfn1_rs::Gfn1Error::InvalidInput(format!("invalid --d4-atm-cutoff `{value}`"))
                })?);
            }
            "--d4-s9" => {
                let value = args.next().ok_or_else(|| {
                    gfn1_rs::Gfn1Error::InvalidInput("--d4-s9 needs a value".to_string())
                })?;
                d4_s9 = Some(value.parse::<f64>().map_err(|_| {
                    gfn1_rs::Gfn1Error::InvalidInput(format!("invalid --d4-s9 `{value}`"))
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
                    gfn1_rs::Gfn1Error::InvalidInput(
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
                        return Err(gfn1_rs::Gfn1Error::InvalidInput(format!(
                            "--multipole-model `{other}` (want mdftb2 | camm_on_mdftb2)"
                        )))
                    }
                }
            }
            "--camm-damp" => {
                multipole = true;
                camm_on_mdftb2 = true;
                let value = args.next().ok_or_else(|| {
                    gfn1_rs::Gfn1Error::InvalidInput(
                        "--camm-damp needs the range factor κ (> 0, e.g. 1.0)".to_string(),
                    )
                })?;
                camm_damp = value.parse::<f64>().map_err(|_| {
                    gfn1_rs::Gfn1Error::InvalidInput(format!("invalid --camm-damp `{value}`"))
                })?;
                camm_damp_set = true;
            }
            "--camm-aes-scale" => {
                multipole = true;
                camm_on_mdftb2 = true;
                let value = args.next().ok_or_else(|| {
                    gfn1_rs::Gfn1Error::InvalidInput(
                        "--camm-aes-scale needs the amplitude s_AES (≥ 0, e.g. 1.0)".to_string(),
                    )
                })?;
                camm_aes_scale = value.parse::<f64>().map_err(|_| {
                    gfn1_rs::Gfn1Error::InvalidInput(format!("invalid --camm-aes-scale `{value}`"))
                })?;
                camm_aes_set = true;
            }
            "--camm-preset" => {
                multipole = true;
                camm_on_mdftb2 = true;
                let value = args.next().ok_or_else(|| {
                    gfn1_rs::Gfn1Error::InvalidInput(
                        "--camm-preset needs a preset name (polar | halogen | halogen-v1 | halogen-allgrad | sigma-hole)".to_string(),
                    )
                })?;
                camm_preset = Some(value);
            }
            "--camm-onsite-scale" => {
                multipole = true;
                camm_on_mdftb2 = true;
                let value = args.next().ok_or_else(|| {
                    gfn1_rs::Gfn1Error::InvalidInput(
                        "--camm-onsite-scale needs the on-site penalty scale s_onsite (≥ 0, e.g. 1.0)"
                            .to_string(),
                    )
                })?;
                camm_onsite_scale = value.parse::<f64>().map_err(|_| {
                    gfn1_rs::Gfn1Error::InvalidInput(format!(
                        "invalid --camm-onsite-scale `{value}`"
                    ))
                })?;
                camm_onsite_set = true;
            }
            "--camm-damp-elem" => {
                multipole = true;
                camm_on_mdftb2 = true;
                let value = args.next().ok_or_else(|| {
                    gfn1_rs::Gfn1Error::InvalidInput(
                        "--camm-damp-elem needs per-element κ, e.g. `N:3.0,O:3.0,C:1.5`".to_string(),
                    )
                })?;
                for tok in value.split(',').filter(|t| !t.trim().is_empty()) {
                    let (sym, kstr) = tok.split_once(':').ok_or_else(|| {
                        gfn1_rs::Gfn1Error::InvalidInput(format!(
                            "invalid --camm-damp-elem entry `{tok}` (want `Elem:κ`)"
                        ))
                    })?;
                    let z = sym.trim().parse::<u8>().ok().or_else(|| gfn1_rs::symbol_to_z(sym.trim()))
                        .ok_or_else(|| {
                            gfn1_rs::Gfn1Error::InvalidInput(format!(
                                "invalid element `{sym}` in --camm-damp-elem"
                            ))
                        })?;
                    let k = kstr.trim().parse::<f64>().map_err(|_| {
                        gfn1_rs::Gfn1Error::InvalidInput(format!("invalid κ `{kstr}` in --camm-damp-elem"))
                    })?;
                    camm_damp_elem.push((z, k));
                }
            }
            "--camm-onsite-scale-elem" => {
                multipole = true;
                camm_on_mdftb2 = true;
                let value = args.next().ok_or_else(|| {
                    gfn1_rs::Gfn1Error::InvalidInput(
                        "--camm-onsite-scale-elem needs per-element s_onsite, e.g. `Cl:0.02,Si:1.0`"
                            .to_string(),
                    )
                })?;
                for tok in value.split(',').filter(|t| !t.trim().is_empty()) {
                    let (sym, sstr) = tok.split_once(':').ok_or_else(|| {
                        gfn1_rs::Gfn1Error::InvalidInput(format!(
                            "invalid --camm-onsite-scale-elem entry `{tok}` (want `Elem:s_onsite`)"
                        ))
                    })?;
                    let z = sym.trim().parse::<u8>().ok().or_else(|| gfn1_rs::symbol_to_z(sym.trim()))
                        .ok_or_else(|| {
                            gfn1_rs::Gfn1Error::InvalidInput(format!(
                                "invalid element `{sym}` in --camm-onsite-scale-elem"
                            ))
                        })?;
                    let s = sstr.trim().parse::<f64>().map_err(|_| {
                        gfn1_rs::Gfn1Error::InvalidInput(format!(
                            "invalid s_onsite `{sstr}` in --camm-onsite-scale-elem"
                        ))
                    })?;
                    camm_onsite_scale_elem.push((z, s));
                }
            }
            "--hubbard-u" => {
                plus_u = true;
                let value = args.next().ok_or_else(|| {
                    gfn1_rs::Gfn1Error::InvalidInput(
                        "--hubbard-u needs per-element U (Hartree), e.g. `Fe:0.15,Ni:0.12`".to_string(),
                    )
                })?;
                for tok in value.split(',').filter(|t| !t.trim().is_empty()) {
                    let (sym, ustr) = tok.split_once(':').ok_or_else(|| {
                        gfn1_rs::Gfn1Error::InvalidInput(format!(
                            "invalid --hubbard-u entry `{tok}` (want `Elem:U`)"
                        ))
                    })?;
                    let z = sym.trim().parse::<u8>().ok().or_else(|| gfn1_rs::symbol_to_z(sym.trim()))
                        .ok_or_else(|| {
                            gfn1_rs::Gfn1Error::InvalidInput(format!("invalid element `{sym}` in --hubbard-u"))
                        })?;
                    let u = ustr.trim().parse::<f64>().map_err(|_| {
                        gfn1_rs::Gfn1Error::InvalidInput(format!("invalid U `{ustr}` in --hubbard-u"))
                    })?;
                    hubbard_u.push((z, u));
                }
            }
            "--hubbard-v" => {
                plus_u = true;
                plus_u_v = true;
                let value = args.next().ok_or_else(|| {
                    gfn1_rs::Gfn1Error::InvalidInput(
                        "--hubbard-v needs element-pair V, e.g. `Fe:N:0.04`".to_string(),
                    )
                })?;
                for tok in value.split(',').filter(|t| !t.trim().is_empty()) {
                    let parts: Vec<&str> = tok.split(':').collect();
                    if parts.len() != 3 {
                        return Err(gfn1_rs::Gfn1Error::InvalidInput(format!(
                            "invalid --hubbard-v entry `{tok}` (want `Elem:Elem:V`)"
                        )));
                    }
                    let zof = |sym: &str| {
                        sym.trim().parse::<u8>().ok().or_else(|| gfn1_rs::symbol_to_z(sym.trim()))
                    };
                    let za = zof(parts[0]).ok_or_else(|| {
                        gfn1_rs::Gfn1Error::InvalidInput(format!("invalid element `{}` in --hubbard-v", parts[0]))
                    })?;
                    let zb = zof(parts[1]).ok_or_else(|| {
                        gfn1_rs::Gfn1Error::InvalidInput(format!("invalid element `{}` in --hubbard-v", parts[1]))
                    })?;
                    let v = parts[2].trim().parse::<f64>().map_err(|_| {
                        gfn1_rs::Gfn1Error::InvalidInput(format!("invalid V `{}` in --hubbard-v", parts[2]))
                    })?;
                    hubbard_v.push((za, zb, v));
                }
            }
            "--hubbard-v-cutoff" => {
                let value = args.next().ok_or_else(|| {
                    gfn1_rs::Gfn1Error::InvalidInput("--hubbard-v-cutoff needs a distance (bohr)".to_string())
                })?;
                hubbard_v_cutoff = Some(value.trim().parse::<f64>().map_err(|_| {
                    gfn1_rs::Gfn1Error::InvalidInput(format!("invalid --hubbard-v-cutoff `{value}`"))
                })?);
            }
            "--camm-damp-charge" => {
                multipole = true;
                camm_on_mdftb2 = true;
                let value = args.next().ok_or_else(|| {
                    gfn1_rs::Gfn1Error::InvalidInput(
                        "--camm-damp-charge needs `κ0,γ`, e.g. `3.0,4.0` (κ_A = κ0/(1+γ·Δq_A²))"
                            .to_string(),
                    )
                })?;
                let (k0s, gs) = value.split_once(',').ok_or_else(|| {
                    gfn1_rs::Gfn1Error::InvalidInput(format!(
                        "invalid --camm-damp-charge `{value}` (want `κ0,γ`)"
                    ))
                })?;
                let k0 = k0s.trim().parse::<f64>().map_err(|_| {
                    gfn1_rs::Gfn1Error::InvalidInput(format!("invalid κ0 `{k0s}`"))
                })?;
                let g = gs.trim().parse::<f64>().map_err(|_| {
                    gfn1_rs::Gfn1Error::InvalidInput(format!("invalid γ `{gs}`"))
                })?;
                camm_damp_charge = Some((k0, g));
            }
            "--multipole-order" => {
                multipole = true;
                let value = args.next().ok_or_else(|| {
                    gfn1_rs::Gfn1Error::InvalidInput(
                        "--multipole-order needs an integer rank (>=4 enables the experimental \
                         arbitrary-rank path; <4 ≡ legacy dipole/quad[/octupole])"
                            .to_string(),
                    )
                })?;
                multipole_order = value.parse::<usize>().map_err(|_| {
                    gfn1_rs::Gfn1Error::InvalidInput(format!("invalid --multipole-order `{value}`"))
                })?;
            }
            "--multipole-charge-order" => {
                multipole = true;
                let value = args.next().ok_or_else(|| {
                    gfn1_rs::Gfn1Error::InvalidInput(
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
                            gfn1_rs::Gfn1Error::InvalidInput(format!(
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
                    gfn1_rs::Gfn1Error::InvalidInput(
                        "--charge-order needs an integer (>=3; 3 = stock GFN1)".to_string(),
                    )
                })?;
                charge_order = value.parse::<usize>().map_err(|_| {
                    gfn1_rs::Gfn1Error::InvalidInput(format!("invalid --charge-order `{value}`"))
                })?;
            }
            "--no-broyden" => no_broyden = true,
            "--gradient" | "--grad" => print_gradient = true,
            "--hessian" | "--hess" => print_hessian = true,
            "--third-derivative" | "--cubic" => print_third = true,
            "--stress" => print_stress = true,
            "--optimize" | "--opt" => optimize = true,
            "--opt-max-iter" => {
                let value = args.next().ok_or_else(|| {
                    gfn1_rs::Gfn1Error::InvalidInput("--opt-max-iter needs a value".to_string())
                })?;
                opt_max_iter = Some(value.parse::<usize>().map_err(|_| {
                    gfn1_rs::Gfn1Error::InvalidInput(format!(
                        "invalid --opt-max-iter value `{value}`"
                    ))
                })?);
            }
            "--opt-gtol" => {
                let value = args.next().ok_or_else(|| {
                    gfn1_rs::Gfn1Error::InvalidInput("--opt-gtol needs a value".to_string())
                })?;
                opt_gtol = Some(value.parse::<f64>().map_err(|_| {
                    gfn1_rs::Gfn1Error::InvalidInput(format!("invalid --opt-gtol value `{value}`"))
                })?);
            }
            "--opt-output" => opt_output = args.next(),
            "--opt-traj" => opt_traj = args.next(),
            "--mixing" => {
                let value = args.next().ok_or_else(|| {
                    gfn1_rs::Gfn1Error::InvalidInput("--mixing needs a value".to_string())
                })?;
                mixing = Some(value.parse::<f64>().map_err(|_| {
                    gfn1_rs::Gfn1Error::InvalidInput(format!("invalid --mixing value `{value}`"))
                })?);
            }
            "--electronic-temperature" | "--etemp" => {
                let value = args.next().ok_or_else(|| {
                    gfn1_rs::Gfn1Error::InvalidInput(
                        "--electronic-temperature needs a value (K)".to_string(),
                    )
                })?;
                electronic_temperature = Some(value.parse::<f64>().map_err(|_| {
                    gfn1_rs::Gfn1Error::InvalidInput(format!(
                        "invalid --electronic-temperature value `{value}`"
                    ))
                })?);
            }
            "--max-scc" => {
                let value = args.next().ok_or_else(|| {
                    gfn1_rs::Gfn1Error::InvalidInput("--max-scc needs a value".to_string())
                })?;
                max_scc = Some(value.parse::<usize>().map_err(|_| {
                    gfn1_rs::Gfn1Error::InvalidInput(format!("invalid --max-scc value `{value}`"))
                })?);
            }
            "--broyden-size" => {
                let value = args.next().ok_or_else(|| {
                    gfn1_rs::Gfn1Error::InvalidInput("--broyden-size needs a value".to_string())
                })?;
                broyden_size = Some(value.parse::<usize>().map_err(|_| {
                    gfn1_rs::Gfn1Error::InvalidInput(format!(
                        "invalid --broyden-size value `{value}`"
                    ))
                })?);
            }
            "--charges" => print_charges = true,
            "--matrices" => print_matrices = true,
            "--charge" => {
                let value = args.next().ok_or_else(|| {
                    gfn1_rs::Gfn1Error::InvalidInput("--charge needs a value".to_string())
                })?;
                charge = Some(value.parse::<f64>().map_err(|_| {
                    gfn1_rs::Gfn1Error::InvalidInput(format!("invalid --charge value `{value}`"))
                })?);
            }
            "--multiplicity" | "--spin-multiplicity" => {
                let value = args.next().ok_or_else(|| {
                    gfn1_rs::Gfn1Error::InvalidInput("--multiplicity needs a value".to_string())
                })?;
                multiplicity = Some(value.parse::<usize>().map_err(|_| {
                    gfn1_rs::Gfn1Error::InvalidInput(format!(
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
                        gfn1_rs::Gfn1Error::InvalidInput(format!(
                            "--field needs three components (Ex Ey Ez); missing component {}",
                            axis + 1
                        ))
                    })?;
                    *slot = value.parse::<f64>().map_err(|_| {
                        gfn1_rs::Gfn1Error::InvalidInput(format!("invalid --field value `{value}`"))
                    })?;
                }
                field = Some(values);
            }
            "--bfield" => {
                let mut values = [0.0f64; 3];
                for (axis, slot) in values.iter_mut().enumerate() {
                    let value = args.next().ok_or_else(|| {
                        gfn1_rs::Gfn1Error::InvalidInput(format!(
                            "--bfield needs three components (Bx By Bz); missing component {}",
                            axis + 1
                        ))
                    })?;
                    *slot = value.parse::<f64>().map_err(|_| {
                        gfn1_rs::Gfn1Error::InvalidInput(format!(
                            "invalid --bfield value `{value}`"
                        ))
                    })?;
                }
                bfield = Some(values);
            }
            "--polarizability" | "--polar" => print_polarizability = true,
            "--ir" => print_ir = true,
            "--raman" => print_raman = true,
            "--tda" => print_tda = true,
            "--tda-grad" => {
                print_tda = true;
                tda_grad = true;
            }
            "--tda-spin" => tda_spin = args.next().unwrap_or_else(|| "singlet".to_string()),
            "--tda-gradient-method" => {
                tda_grad_method = args.next().unwrap_or_else(|| "semi-numerical".to_string())
            }
            "--nmr" => {
                let value = args.next().ok_or_else(|| {
                    gfn1_rs::Gfn1Error::InvalidInput(
                        "--nmr needs a nucleus index (0-based)".to_string(),
                    )
                })?;
                nmr_nucleus = Some(value.parse::<usize>().map_err(|_| {
                    gfn1_rs::Gfn1Error::InvalidInput(format!("invalid --nmr value `{value}`"))
                })?);
            }
            "--tda-state" => {
                let value = args.next().ok_or_else(|| {
                    gfn1_rs::Gfn1Error::InvalidInput("--tda-state needs a value".to_string())
                })?;
                tda_state = value.parse::<usize>().map_err(|_| {
                    gfn1_rs::Gfn1Error::InvalidInput(format!("invalid --tda-state value `{value}`"))
                })?;
            }
            "--tda-step" => {
                let value = args.next().ok_or_else(|| {
                    gfn1_rs::Gfn1Error::InvalidInput("--tda-step needs a value".to_string())
                })?;
                tda_step = value.parse::<f64>().map_err(|_| {
                    gfn1_rs::Gfn1Error::InvalidInput(format!("invalid --tda-step value `{value}`"))
                })?;
            }
            "--n-states" => {
                let value = args.next().ok_or_else(|| {
                    gfn1_rs::Gfn1Error::InvalidInput("--n-states needs a value".to_string())
                })?;
                tda_n_states = value.parse::<usize>().map_err(|_| {
                    gfn1_rs::Gfn1Error::InvalidInput(format!("invalid --n-states value `{value}`"))
                })?;
            }
            "--scc-accel" => scc_accel = args.next(),
            "--level-shift" => {
                let value = args.next().ok_or_else(|| {
                    gfn1_rs::Gfn1Error::InvalidInput("--level-shift needs a value".to_string())
                })?;
                level_shift = value.parse::<f64>().map_err(|_| {
                    gfn1_rs::Gfn1Error::InvalidInput(format!(
                        "invalid --level-shift value `{value}`"
                    ))
                })?;
            }
            "--field-step" => {
                let value = args.next().ok_or_else(|| {
                    gfn1_rs::Gfn1Error::InvalidInput("--field-step needs a value".to_string())
                })?;
                field_step = value.parse::<f64>().map_err(|_| {
                    gfn1_rs::Gfn1Error::InvalidInput(format!(
                        "invalid --field-step value `{value}`"
                    ))
                })?;
            }
            "--param-deriv" => param_deriv = true,
            "--all-param-targets" => all_param_targets = true,
            "--param-forces" => param_forces = true,
            "--param-stress" => param_stress = true,
            "--target-chunk" => target_chunk = args.next(),
            "--targets" => param_targets = args.next(),
            "--param-step" => {
                let value = args.next().ok_or_else(|| {
                    gfn1_rs::Gfn1Error::InvalidInput("--param-step needs a value".to_string())
                })?;
                param_step = value.parse::<f64>().map_err(|_| {
                    gfn1_rs::Gfn1Error::InvalidInput(format!(
                        "invalid --param-step value `{value}`"
                    ))
                })?;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other if xyz.is_none() => xyz = Some(other.to_string()),
            other => {
                return Err(gfn1_rs::Gfn1Error::InvalidInput(format!(
                    "unexpected argument `{other}`"
                )));
            }
        }
    }

    let xyz =
        xyz.ok_or_else(|| gfn1_rs::Gfn1Error::InvalidInput("missing XYZ file".to_string()))?;
    let params = Gfn1Parameters::resolve(param.as_deref())?;
    println!("parameters: {}", params.source_description());
    let system = PeriodicSystem::from_xyz_file(&xyz, charge.unwrap_or(0.0), bohr)?;
    let mut options = ElectronicOptions::default();
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
        options.external_field = gfn1_rs::ExternalFieldOptions::electric(Vec3::new(ex, ey, ez));
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
        gfn1_rs::MultipoleModel::CammOnMdftb2
    } else {
        gfn1_rs::MultipoleModel::Mdftb2
    };
    // Resolve a named CAMM preset, filling only the knobs the user did not set explicitly
    // (explicit `--camm-damp*` / `--camm-aes-scale` / `--camm-onsite-scale` always win).
    if let Some(name) = &camm_preset {
        let (gk, elems, aes, onsite, onsite_elem) = gfn1_rs::camm_preset(name).ok_or_else(|| {
            gfn1_rs::Gfn1Error::InvalidInput(format!(
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
        let sec = if let Some(res) = gfn1_rs::builtin_secondary(name) {
            res?
        } else {
            let text = std::fs::read_to_string(name).map_err(|e| {
                gfn1_rs::Gfn1Error::InvalidInput(format!(
                    "--multipole-secondary-basis `{name}` is not a built-in name and cannot be \
                     read as a file: {e}"
                ))
            })?;
            gfn1_rs::parse_secondary_basis(&text)?
        };
        options.multipole_secondary_basis = Some(sec);
    }
    options.charge_order = charge_order;
    if let Some(name) = scc_accel.as_deref() {
        options.scc_accelerator = match name.to_ascii_lowercase().as_str() {
            "broyden" => SccAccelerator::Broyden,
            "linear" => SccAccelerator::Linear,
            "cdiis" => SccAccelerator::Cdiis,
            "newton" => SccAccelerator::Newton,
            other => {
                return Err(gfn1_rs::Gfn1Error::InvalidInput(format!(
                    "unknown --scf-accel `{other}` (use broyden/linear/cdiis/newton)"
                )))
            }
        };
    }
    if let Some([bx, by, bz]) = bfield {
        options.external_field.magnetic_field = Some(Vec3::new(bx, by, bz));
        let result = gfn1_rs::run_magnetic_scc(&system, &params, &options)?;
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
            return Err(gfn1_rs::Gfn1Error::InvalidInput(
                "NMR shielding is available for non-periodic systems only".to_string(),
            ));
        }
        if nucleus >= system.atoms.len() {
            return Err(gfn1_rs::Gfn1Error::InvalidInput(format!(
                "--nmr nucleus index {nucleus} out of range ({} atoms)",
                system.atoms.len()
            )));
        }
        options.external_field.magnetic_field = Some(Vec3::zero());
        let gauge = system.atoms[nucleus].position; // common gauge origin at the nucleus
        let sh = gfn1_rs::nmr_shielding_tensor(&system, &params, &options, None, nucleus, gauge)?;
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
    if print_tda {
        let spin = match tda_spin.to_ascii_lowercase().as_str() {
            "singlet" | "s" => TdaSpin::Singlet,
            "triplet" | "t" => TdaSpin::Triplet,
            other => {
                return Err(gfn1_rs::Gfn1Error::InvalidInput(format!(
                    "unknown --tda-spin `{other}` (use singlet or triplet)"
                )))
            }
        };
        let tda_options = TdaOptions {
            n_states: tda_n_states.max(tda_state + 1),
            spin,
        };
        println!("tda_spin          {:>24}", spin.label());
        if system.lattice.is_none() {
            let electronic = run_electronic(&system, &params, options.clone())?;
            let result = solve_tda(&system, &params, &electronic, tda_options)?;
            print_tda_states(&result);
        } else {
            let result = gfn1_rs::solve_tda_pbc_gamma(&system, &params, &options, tda_options)?;
            println!("(Gamma-point periodic TDA)");
            print_tda_states(&result);
        }
        if tda_grad {
            let mut method = TdaGradientMethod::parse(&tda_grad_method).ok_or_else(|| {
                gfn1_rs::Gfn1Error::InvalidInput(format!(
                    "invalid --tda-gradient-method `{tda_grad_method}` \
                     (expected semi-numerical, fd, or analytic)"
                ))
            })?;
            // Periodic (Gamma-point) systems support the finite-difference and the
            // fully analytic paths; the semi-numerical hybrid is non-periodic, so
            // fall back to finite difference for it.
            if system.lattice.is_some() && method == TdaGradientMethod::SemiNumerical {
                println!("(periodic system: semi-numerical unavailable, forcing --tda-gradient-method fd)");
                method = TdaGradientMethod::FiniteDifference;
            }
            let g = solve_tda_gradient_method(
                &system,
                &params,
                &options,
                tda_state,
                tda_options,
                tda_step,
                method,
            )?;
            println!("tda_state         {:24}", tda_state + 1);
            println!("tda_gradient_method  {tda_grad_method}");
            println!("tda_total_energy  {:24.14}", g.total_energy);
            println!("tda_gradient (state {}, Hartree/bohr)", tda_state + 1);
            for gi in &g.gradient {
                println!("{:24.14e} {:24.14e} {:24.14e}", gi.x, gi.y, gi.z);
            }
        }
        return Ok(());
    }
    if print_polarizability || print_ir || print_raman {
        if system.lattice.is_some() {
            return Err(gfn1_rs::Gfn1Error::InvalidInput(
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
            let raman = raman_spectrum(&system, &params, hess_options, Vec3::zero(), field_step)?;
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
                gfn1_rs::Gfn1Error::InvalidInput(
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
                gfn1_rs::Gfn1Error::InvalidInput(
                    "--target-chunk must be `i/n` (1-based)".to_string(),
                )
            })?;
            let idx = idx.trim().parse::<usize>().map_err(|_| {
                gfn1_rs::Gfn1Error::InvalidInput(format!("invalid chunk index `{idx}`"))
            })?;
            let count = count.trim().parse::<usize>().map_err(|_| {
                gfn1_rs::Gfn1Error::InvalidInput(format!("invalid chunk count `{count}`"))
            })?;
            targets = gfn1_rs::select_target_chunk(targets, idx, count)?;
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
        return Ok(());
    }
    if print_gradient {
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
    if print_third {
        if system.lattice.is_some() {
            return Err(gfn1_rs::Gfn1Error::InvalidInput(
                "third derivative (cubic force constants) supports non-periodic systems only"
                    .to_string(),
            ));
        }
        let hess_options = AnalyticHessianOptions {
            electronic_options: options.clone(),
            ..AnalyticHessianOptions::default()
        };
        let cutoff = options.hamiltonian.coordination_cutoff;
        let slabs = gfn1_rs::third_derivative_analytic(&system, &params, hess_options, cutoff)?;
        // slabs[c][(a,b)] = T_abc = ∂³E/∂R_a∂R_b∂R_c (strict closed form).
        println!("third_derivative_size  {:24}", slabs.len());
        if print_matrices {
            for (c, slab) in slabs.iter().enumerate() {
                print_matrix(&format!("third_derivative_slab[c={c}]"), slab);
            }
        }
        return Ok(());
    }
    if print_stress {
        if system.lattice.is_none() {
            return Err(gfn1_rs::Gfn1Error::InvalidInput(
                "stress requires a periodic input with Lattice/PBC".to_string(),
            ));
        }
        let pbc = PbcOptions::for_boundary(options.boundary);
        let result = pbc_stress(&system, &params, &options, &pbc)?;
        print_energy_terms(&gfn1_rs::pbc_electronic_result(
            result.scf.clone(),
            &system,
            pbc.ao_cutoff,
        )?);
        print_matrix("stress", &result.stress);
        return Ok(());
    }
    let result = run_electronic(&system, &params, options)?;
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

fn print_help() {
    println!(
        r#"Usage:
  gfn1_rs_cli [OPTIONS] molecule.xyz

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
  --scc-accel broyden|linear|cdiis|newton
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
  --gradient, --grad              Print analytic nuclear gradient.
  --hessian, --hess               Print analytic Hessian.
  --third-derivative, --cubic     Print analytic third derivative.
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

TDA:
  --tda                           Solve TDA excited states.
  --n-states N                    Number of TDA states.
  --tda-spin singlet|triplet      TDA spin block.
  --tda-grad                      Print excited-state gradient.
  --tda-state K                   0-based TDA state for --tda-grad.
  --tda-step H                    TDA finite-difference step.
  --tda-gradient-method METHOD    semi-numerical|finite-difference|analytic.

Parameter derivatives:
  --param-deriv                   Print finite-difference parameter derivatives.
  --targets SPEC                  Target list: glob:ks, elem:1:GAM, pair:1:6, ...
  --all-param-targets             Use all active parameter targets.
  --param-forces                  Include force derivatives.
  --param-stress                  Include stress derivatives.
  --target-chunk I/N              Evaluate only chunk I of N target chunks.
  --param-step H                  Parameter finite-difference step.

Other:
  -h, --help                      Show this help.
"#
    );
}

fn print_energy_terms(result: &gfn1_rs::ElectronicResult) {
    println!("total_free        {:24.14}", result.total_free);
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
    let bohr_to_angstrom = 1.0 / gfn1_rs::system::ANGSTROM_TO_BOHR;
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
        let symbol = gfn1_rs::z_to_symbol(atom.z).unwrap_or("X");
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

fn print_matrix(name: &str, matrix: &gfn1_rs::linalg::Matrix) {
    println!("{name} {} {}", matrix.rows(), matrix.cols());
    for i in 0..matrix.rows() {
        for j in 0..matrix.cols() {
            print!(" {:24.16e}", matrix[(i, j)]);
        }
        println!();
    }
}

fn print_tda_states(result: &gfn1_rs::TdaResult) {
    println!(
        "{:>5} {:>18} {:>18} {:>18}",
        "state", "energy_eV", "energy_Hartree", "osc_strength"
    );
    for (idx, st) in result.states.iter().enumerate() {
        println!(
            "{:>5} {:>18.6} {:>18.10} {:>18.8}",
            idx + 1,
            st.excitation_energy * gfn1_rs::constants::HARTREE_TO_EV,
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
