// SPDX-License-Identifier: GPL-3.0-or-later
//! Complete native Rust sTDA-xTB, independent of TD-GFN1 and external programs.
//! Orbital generation, Gaussian integrals, response, selection and gradients run
//! entirely in Rust. Upstream program output is used only as offline test data.
use sha2::{Digest, Sha256};
mod dual;
mod gradient;
mod ground;
mod native;
mod thermal;
pub use gradient::{
    solve_stda_xtb_gradient, solve_stda_xtb_hessian, solve_stda_xtb_total_gradient,
    solve_stda_xtb_total_hessian, NativeStdaGradientResult, NativeStdaHessianResult,
    NativeStdaTotalGradientResult, NativeStdaTotalHessianResult,
};
pub use ground::{
    ground_state as stda_xtb_orbitals,
    ground_state_at_temperature as stda_xtb_orbitals_at_temperature, Orbitals as StdaXtbOrbitals,
};
pub use native::{
    solve_stda_xtb_native, solve_stda_xtb_native as solve_stda_xtb, NativeStdaOptions,
    NativeStdaResult, NativeStdaState,
};
pub const XTB4STDA_COMMIT: &str = "238db6c7df81d5a2ed829f4f682f00aaa87e137d";
pub const STD2_REFERENCE_COMMIT: &str = "8d50f9f2a4b66b534e19bb8a99524be6b7153a47";
const PARAM1: &[u8] = include_bytes!("../third_party/stda_xtb/.param_stda1.xtb");
const PARAM2: &[u8] = include_bytes!("../third_party/stda_xtb/.param_stda2.xtb");
pub fn parameter_sha256() -> [String; 2] {
    [
        format!("{:x}", Sha256::digest(PARAM1)),
        format!("{:x}", Sha256::digest(PARAM2)),
    ]
}
