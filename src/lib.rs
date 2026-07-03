// SPDX-License-Identifier: GPL-3.0-or-later
//! GFN1-xTB components.
//!
//! Parameter files are deliberately external. Use [`resolve_param_path`] or
//! [`Gfn1Parameters::from_file`] with xTB's `param_gfn1-xtb.txt`.

pub mod basis;
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
pub mod gradient;
pub mod halogen;
pub mod hamiltonian;
pub mod hessian;
pub mod integrals;
pub mod lattice;
pub mod linalg;
pub mod magnetic;
pub mod math;
pub mod model;
pub mod multipole;
pub mod nmr;
pub mod plus_u;
pub mod plus_u_dudr;
pub mod optimizer;
pub mod pairlist;
pub mod param_deriv;
pub mod params;
pub mod pbc;
pub mod profile;
pub mod properties;
pub mod repulsion;
pub mod secondary_bases;
pub mod secondary_basis;
pub mod spin;
pub mod sto;
pub mod system;
pub mod td;
pub mod third_derivative;
pub mod trah;
pub mod vibrational;

pub use basis::{AOBasisFunction, BasisOptions, BasisSet, BasisShell};
pub use dispersion::{dispersion_third_derivative, DispersionThirdResult};
pub use electronic::{
    camm_preset, run_electronic, run_electronic_rank_ladder, ElectronicOptions, ElectronicResult,
    EnergyTerms, Gfn1Calculator, MultipoleModel, SccAccelerator,
};
pub use error::{Gfn1Error, Result};
pub use field::{mulliken_dipole, ExternalFieldOptions};
pub use gradient::{analytic_gradient, AnalyticGradientOptions, AnalyticGradientResult};
pub use hessian::{
    analytic_hessian, analytic_hessian_from_result, analytic_repulsion_hessian,
    fixed_density_cn_h0_hessian, fixed_density_cn_h0_third_derivative, fixed_density_pulay_hessian,
    fixed_shell_charge_scc_hessian, AnalyticHessianOptions, AnalyticHessianResult,
    FixedDensityCnH0HessianResult, FixedDensityPulayHessianResult, FixedSccHessianResult,
};
pub use magnetic::{
    angular_momentum_matrix, cotton_mouton_tensor, lao_dipole_matrix, magnetic_analytic_gradient,
    magnetic_gradient, magnetic_h0_overlap, magnetic_polarizability,
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
    resolve_param_path, AngularMomentum, ElementParam, Gfn1Parameters, ParameterTarget, ShellParam,
    GFN1_D3_REFERENCE_ENV,
};
pub use pbc::{
    pbc_analytic_gradient, pbc_electronic_result, pbc_gamma_hessian, pbc_kpoint_hessian,
    pbc_stress, run_electronic_pbc, run_pbc_scc, EwaldOptions, KMesh, PbcGradientResult,
    PbcHessianResult, PbcOptions, PbcSccResult, PbcStressResult,
};
pub use properties::{
    dipole_derivatives, ir_spectrum, polarizability_derivatives, raman_spectrum,
    static_polarizability, static_polarizability_finite_field, DipoleDerivatives, IrMode,
    IrSpectrum, Polarizability, PolarizabilityDerivatives, RamanMode, RamanSpectrum,
};
pub use secondary_bases::{builtin_secondary, builtin_secondary_text};
pub use secondary_basis::{parse_secondary_basis, SecondaryBasis};
pub use system::{symbol_to_z, z_to_symbol, Atom, Molecule, PeriodicSystem};
pub use td::{
    solve_tda, solve_tda_gradient, solve_tda_gradient_analytic, solve_tda_gradient_method,
    solve_tda_gradient_seminumerical, solve_tda_kpoint, solve_tda_kpoint_gradient,
    solve_tda_kpoint_gradient_analytic, solve_tda_pbc_gamma, tda_frozen_excitation_energy,
    tda_optical_rotation, tda_rotatory_strengths, RotatoryState, TdaGradientMethod,
    TdaGradientResult, TdaOptions, TdaResult, TdaSpin, TdaState,
};
pub use third_derivative::{
    third_derivative_analytic, third_derivative_analytic_block, third_derivative_analytic_dense,
    third_derivative_analytic_vector, third_derivative_dispersion, third_derivative_frozen,
    third_derivative_frozen_complete, third_derivative_frozen_full, third_derivative_geometric,
    third_derivative_seminumerical_block, third_derivative_seminumerical_dense,
    third_derivative_seminumerical_vector, SymmetricThird,
};
pub use vibrational::{vibrational_analysis, VibrationalModes};

#[cfg(feature = "python")]
pub mod python;
