// SPDX-License-Identifier: GPL-3.0-or-later
//! GFN1-xTB components.
//!
//! The official GFN1-xTB parameter files are BUNDLED (embedded from
//! `third_party/xtb`, LGPL-3.0-or-later): [`Gfn1Parameters::resolve`]`(None)`
//! resolves explicit path > `GFN1_XTB_PARAM` > builtin, so the crate works
//! with no external files. [`Gfn1Parameters::from_file`] still loads any
//! compatible parameter file.

pub mod basis;
pub mod citations;
pub mod cli;
pub mod constants;
pub mod coordination;
pub mod coulomb;
pub mod cphf;
pub mod d4_reference;
pub mod data_tables;
pub mod dispersion;
pub mod electronic;
pub mod error;
pub mod exchange;
pub mod field;
pub mod fourth_derivative;
pub mod fragment;
pub mod gradient;
pub mod halogen;
pub mod hamiltonian;
pub mod hessian;
pub mod integrals;
pub mod jets;
pub mod lattice;
pub mod linalg;
pub mod logging;
pub mod magnetic;
pub mod math;
pub mod ml;
mod variational_derivatives;
mod derivative_assembly;
mod derivative_files;
pub use derivative_files::{
    fourth_derivative_with_storage, third_derivative_with_storage, DerivativeOutput,
    DerivativeStorageOptions, PackedDerivativeFile, PackedDerivativeMetadata,
};
pub mod model;
pub mod multipole;
pub mod nmr;
pub mod optimizer;
pub mod pairlist;
pub mod param_deriv;
pub mod params;
pub mod pbc;
pub mod plus_u;
pub mod plus_u_dudr;
pub mod profile;
pub mod properties;
pub mod property_training;
pub mod repulsion;
pub mod resources;
pub mod response;
pub(crate) mod scalar_dual;
pub mod secondary_bases;
pub mod secondary_basis;
pub mod smoothness;
pub mod soc;
pub mod solvation;
pub mod spin;
pub mod stda_xtb;
pub mod sto;
pub mod system;
pub mod td;
pub mod terms;
pub mod third_derivative;
pub mod trah;
pub mod training;
pub mod vibrational;
pub mod vpt2;

pub use basis::{AOBasisFunction, BasisOptions, BasisSet, BasisShell};
pub use dispersion::{
    dispersion_fourth_derivative, dispersion_third_derivative, DispersionFourthResult,
    DispersionThirdResult, MAX_FOURTH_DERIVATIVE_NDOF,
};
pub use electronic::{
    camm_preset, run_electronic, run_electronic_rank_ladder, ElectronicOptions, ElectronicResult,
    EnergyTerms, Gfn1Calculator, MultipoleModel, SccAccelerator,
};
pub use error::{Gfn1Error, Result};
pub use field::{mulliken_dipole, ExternalFieldOptions};
pub use fourth_derivative::{
    directional_fourth_derivative, directional_fourth_seminumerical,
    fourth_derivative_analytic_block, fourth_derivative_analytic_block_with_limits,
    fourth_derivative_analytic_dense, fourth_derivative_analytic_semidiagonal,
    QuarticAssemblyOptions, QuarticReference, SymmetricFourth,
};
pub use gradient::{analytic_gradient, AnalyticGradientOptions, AnalyticGradientResult};
pub use hessian::{
    analytic_hessian, analytic_hessian_from_result, analytic_repulsion_hessian,
    fixed_density_cn_h0_hessian, fixed_density_cn_h0_third_derivative, fixed_density_pulay_hessian,
    fixed_shell_charge_scc_hessian, AnalyticHessianOptions, AnalyticHessianResult,
    FixedDensityCnH0HessianResult, FixedDensityPulayHessianResult, FixedSccHessianResult,
};
pub use logging::{CalculationLogger, LogLevel};
pub use magnetic::{
    angular_momentum_matrix, boosted_overlap_pair, cotton_mouton_tensor, lao_dipole_matrix,
    magnetic_analytic_gradient, magnetic_gradient, magnetic_h0_overlap, magnetic_polarizability,
    magnetizability_diagonal_analytic, magnetizability_isotropic,
    magnetizability_isotropic_analytic, magnetizability_tensor_analytic, mcd_tensor,
    nmr_shielding_tensor, run_magnetic_scc, run_magnetic_scc_m1, MagneticGradientResult,
    MagneticSccResult, NmrShielding, MAGNETIZABILITY_AU_TO_SI, SPEED_OF_LIGHT_AU,
};
pub use optimizer::{
    optimize_geometry, GeometryOptimizationOptions, GeometryOptimizationResult,
    GeometryOptimizationStep,
};
pub use param_deriv::{
    active_targets_for_system, parameter_dipole_derivatives, parameter_finite_difference,
    parameter_hessian_derivatives, select_target_chunk, ParamDerivative, ParamDerivativeOptions,
};
pub use params::{
    resolve_param_path, AngularMomentum, ElementParam, Gfn1Parameters, ParamSource,
    ParameterTarget, ShellParam, BUILTIN_PARAM_PROVENANCE, GFN1_D3_REFERENCE_ENV, GFN1_PARAM_ENV,
};
pub use pbc::{
    pbc_analytic_gradient, pbc_berry_polarization, pbc_electronic_result, pbc_gamma_hessian,
    pbc_gamma_third_analytic_block, pbc_gamma_third_analytic_dense,
    pbc_gamma_third_analytic_vector, pbc_gamma_third_with_reference, pbc_gruneisen,
    pbc_kpoint_hessian, pbc_kpoint_strain_hessian_derivative, pbc_kpoint_third_analytic_block,
    pbc_kpoint_third_analytic_dense, pbc_kpoint_third_analytic_vector,
    pbc_kpoint_third_derivative_seminumerical_dense,
    pbc_kpoint_third_derivative_seminumerical_vector, pbc_kpoint_third_with_reference,
    pbc_strain_hessian_derivative, pbc_stress, pbc_third_derivative_seminumerical_dense,
    pbc_third_derivative_seminumerical_vector, run_electronic_pbc, run_pbc_scc,
    scale_lattice_isotropic, BerryMethodSelector, BerryPolarizationMethod,
    BerryPolarizationOptions, BerryPolarizationResult, EwaldOptions, GammaThirdReference,
    GruneisenOptions, GruneisenResult, KMesh, KpointThirdReference, PbcGradientResult,
    PbcHessianResult, PbcOptions, PbcSccResult, PbcStressResult, SecondOrderStencil,
    POLARIZATION_AU_TO_C_PER_M2,
};
pub use properties::{
    dipole_derivatives, ir_spectrum, polarizability_derivatives, raman_spectrum,
    static_polarizability, static_polarizability_finite_field, DipoleDerivatives, IrMode,
    IrSpectrum, Polarizability, PolarizabilityDerivatives, RamanMode, RamanSpectrum,
};
pub use secondary_bases::{builtin_secondary, builtin_secondary_text};
pub use secondary_basis::{parse_secondary_basis, SecondaryBasis};
pub use solvation::{
    SolutionState, SolvationCache, SolvationEnergy, SolvationModel, SolvationOptions,
};
pub use system::{symbol_to_z, z_to_symbol, Atom, Molecule, PeriodicSystem};
pub use td::{
    solve_optical_td_dfpt, solve_rpa, solve_rpa_gradient_analytic, solve_rpa_hessian_analytic, solve_rpa_manifold_gradient_analytic,
    solve_rpa_manifold_hessian_analytic, OpticalTdDfptResult,
    solve_tda, solve_tda_gradient, solve_tda_gradient_analytic, solve_tda_gradient_method,
    solve_tda_gradient_seminumerical, solve_tda_hessian_analytic,
    solve_tda_hessian_analytic_with_budget, solve_tda_kpoint, solve_tda_kpoint_gradient,
    solve_tda_kpoint_gradient_analytic, solve_tda_manifold_gradient, solve_tda_manifold_hessian_analytic, solve_tda_manifold_hessian_analytic_with_budget, solve_tda_pbc_gamma, solve_tda_with_dipole,
    tda_frozen_excitation_energy, tda_optical_rotation, tda_rotatory_strengths, RotatoryState,
    TdaGradientMethod, TdaGradientResult, TdaHessianResult, TdaManifoldGradientResult, TdaManifoldHessianResult, TdaOptions, TdaResult, TdaSpin,
    TdaState, TdaTransitionDipole,
};
pub use third_derivative::finite_t::{
    directional_fourth_finite_t, directional_third_finite_t, fourth_derivative_finite_t_block,
    fourth_derivative_finite_t_dense, third_derivative_finite_t_block,
    third_derivative_finite_t_dense, third_derivative_finite_t_vector, FiniteTThirdReference,
};
pub use third_derivative::{
    third_derivative_analytic, third_derivative_analytic_block, third_derivative_analytic_dense,
    third_derivative_analytic_vector, third_derivative_dispersion, third_derivative_frozen,
    third_derivative_frozen_complete, third_derivative_frozen_full, third_derivative_geometric,
    third_derivative_seminumerical_block, third_derivative_seminumerical_dense,
    third_derivative_seminumerical_vector, SymmetricThird,
};
pub use vibrational::{
    molecular_vibrational_analysis, vibrational_analysis, MolecularVibrationOptions,
    MolecularVibrations, VibrationalModes,
};
pub use vpt2::{
    molecular_vpt2_pes, vpt2_pes, MolecularVpt2Result, NormalModeForceField, Vpt2Options,
    Vpt2Result,
};

#[cfg(feature = "python")]
pub mod python;
#[cfg(feature = "python")]
mod python_resources;
