// SPDX-License-Identifier: GPL-3.0-or-later
//! Seminumerical periodic nuclear third derivatives — Gamma-point **and**
//! Brillouin-zone-sampled (k-point) — plus the strain-mixed derivative
//! `dH/d(ln V)` that feeds the Grueneisen module.
//!
//! Everything here is a **central finite difference of an analytic PBC Hessian**
//! ([`crate::pbc::hessian::pbc_gamma_hessian`] for the Gamma entry points,
//! [`crate::pbc::hessian::pbc_kpoint_hessian`] for the `pbc_kpoint_*` ones): each
//! `+`/`-` displacement re-runs the full periodic SCC and the full analytic
//! Hessian, so the result inherits the (FD-validated) analytic Hessian rather
//! than double-differencing energies. There is deliberately no closed-form
//! periodic third derivative in v0.5.0: the periodic CPXTB response chain would
//! have to be differentiated once more through the Ewald/KO partitioning, which
//! is a separate work item — and the k-point entry points here are explicitly the
//! **verification base** for that future analytic k-point FC3.
//!
//! # Gamma versus k-point
//!
//! The two families are the *same* finite difference over the *same* atomic
//! displacements; they differ only in which analytic Hessian is differentiated,
//! and therefore in what they mean physically:
//!
//! * The Gamma entry points give `d^3 E / dR^3` of a **supercell treated as a
//!   molecule with periodic images** — the Brillouin zone is sampled at one point,
//!   so the cubic force constants are those of the `q = 0` dynamical matrix only.
//! * The `pbc_kpoint_*` entry points differentiate the k-point Hessian, i.e. the
//!   `q = 0` cubic force constants of a **Brillouin-zone-converged** electronic
//!   structure. The k-mesh is taken from [`PbcOptions::kmesh`], exactly as
//!   [`crate::pbc::hessian::pbc_kpoint_hessian`] takes it. They are *not* phonon
//!   FC3s at finite `q`: the nuclear displacement pattern is still one atom of
//!   one cell, so what converges with the mesh is the electronic BZ sum, not the
//!   phonon wavevector.
//!
//! A `1 x 1 x 1` mesh makes the k-point path reduce to the Gamma path (the
//! complex CPXTB collapses onto the real one); the two then agree to the
//! iterative-solver noise of the k-point CPXTB rather than bit-for-bit — see the
//! integration gate `kpoint_third_derivative_gamma_mesh_matches_gamma_path`.
//!
//! # Conventions
//!
//! * **Nuclear displacements** are *absolute Cartesian* shifts of one atom
//!   (Bohr), with the lattice held fixed — the same convention the PBC Hessian's
//!   own finite-difference gates use (`shift()` in `pbc::hessian`'s test module
//!   and `displace()` in `tests/hessian.rs`). Fractional coordinates therefore
//!   change; the cell does not.
//! * **Volumetric strain** is *isotropic frozen-ion* scaling: the three lattice
//!   vectors are multiplied by `(1 +/- delta)^(1/3)` so that `V -> (1 +/- delta) V`,
//!   and the atoms follow with **frozen fractional coordinates** (which under an
//!   isotropic scaling is exactly `r -> s r`). No internal-coordinate relaxation
//!   is performed — see [`crate::pbc::gruneisen`] for why that is the standard
//!   convention for mode Grueneisen parameters, and for the relaxed-ion variant
//!   left as future work.
//!
//! # Cost
//!
//! `pbc_third_derivative_seminumerical_dense` needs `2 * 3N` analytic periodic
//! Hessians; the vector mode needs `2 * nnz(v)`; the strain derivative needs
//! exactly `2`. The `pbc_kpoint_*` variants need the same *count*, each roughly
//! `n_k` times dearer (and complex), so the mesh — not the displacement sweep —
//! sets the bill.
//!
//! # Caveat: finite electronic temperature
//!
//! Everything gated for these entry points is at `electronic_temperature = 0`
//! (integer occupations, gapped insulators), and that is the regime to trust
//! them in. The periodic finite-temperature response itself is a direct
//! dielectric solve as of v0.5.0 (it verifies its own residual rather than
//! running an unchecked fixed point), so a smeared run is no longer *unsound* —
//! but nothing here differences a smeared periodic system in anger. Two reasons
//! to stay careful if you do:
//!
//! * every estimator here differences a **reconverged** quantity, so metallic
//!   SCC reconvergence noise enters divided by `2 * step`; tighten
//!   `charge_tolerance` / `energy_tolerance` well past their defaults first;
//! * band reordering between the `+`/`-` geometries makes the differenced
//!   quantity non-smooth, which is a property of the fixture rather than of the
//!   response.

use crate::electronic::ElectronicOptions;
use crate::error::{Gfn1Error, Result};
use crate::lattice::Lattice;
use crate::linalg::Matrix;
use crate::math::Mat3;
use crate::params::Gfn1Parameters;
use crate::pbc::hessian::{pbc_gamma_hessian, pbc_kpoint_hessian, PbcHessianResult};
use crate::pbc::PbcOptions;
use crate::system::PeriodicSystem;

/// The analytic periodic Hessian a seminumerical derivative differentiates.
///
/// [`pbc_gamma_hessian`] and [`pbc_kpoint_hessian`] share this signature exactly,
/// so the Gamma and k-point entry points below are the same finite-difference
/// machinery instantiated on one or the other — there is no second copy of the
/// displacement, cutoff or strain bookkeeping.
type HessianEvaluator = fn(
    &PeriodicSystem,
    &Gfn1Parameters,
    &ElectronicOptions,
    &PbcOptions,
) -> Result<PbcHessianResult>;

/// Shift one Cartesian degree of freedom (`dof = 3*atom + axis`) by `step` Bohr,
/// leaving the lattice untouched. Mirrors the displacement convention of the
/// existing PBC Hessian finite-difference gates.
fn shift(system: &mut PeriodicSystem, dof: usize, step: f64) {
    let atom = dof / 3;
    match dof % 3 {
        0 => system.atoms[atom].position.x += step,
        1 => system.atoms[atom].position.y += step,
        _ => system.atoms[atom].position.z += step,
    }
}

fn require_periodic(system: &PeriodicSystem, who: &str) -> Result<Lattice> {
    system.lattice.ok_or_else(|| {
        Gfn1Error::InvalidInput(format!("{who}: the system has no lattice (not periodic)"))
    })
}

fn require_positive_step(step: f64, who: &str) -> Result<()> {
    if !(step.is_finite() && step > 0.0) {
        return Err(Gfn1Error::InvalidInput(format!(
            "{who}: the finite-difference step must be finite and positive (got {step})"
        )));
    }
    Ok(())
}

/// One slab `dH_ab/dR_c` from a central difference along DOF `c`, with the
/// analytic Hessian supplied by the caller (Gamma or k-point).
fn hessian_slab_with(
    hessian: HessianEvaluator,
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    dof: usize,
    step: f64,
) -> Result<Matrix> {
    let ndof = 3 * system.atoms.len();
    let mut plus = system.clone();
    let mut minus = system.clone();
    shift(&mut plus, dof, step);
    shift(&mut minus, dof, -step);
    let hp = hessian(&plus, params, options, pbc)?.hessian;
    let hm = hessian(&minus, params, options, pbc)?.hessian;
    let scale = 1.0 / (2.0 * step);
    let mut slab = Matrix::zeros(ndof, ndof);
    for a in 0..ndof {
        for b in 0..ndof {
            slab[(a, b)] = (hp[(a, b)] - hm[(a, b)]) * scale;
        }
    }
    Ok(slab)
}

/// Shared implementation of the dense seminumerical third derivative.
fn seminumerical_dense_with(
    hessian: HessianEvaluator,
    who: &str,
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    step: f64,
) -> Result<Vec<Matrix>> {
    require_periodic(system, who)?;
    require_positive_step(step, who)?;
    let ndof = 3 * system.atoms.len();
    let mut slabs = Vec::with_capacity(ndof);
    for c in 0..ndof {
        slabs.push(hessian_slab_with(
            hessian, system, params, options, pbc, c, step,
        )?);
    }
    Ok(slabs)
}

/// Shared implementation of the directional (vector) seminumerical third
/// derivative.
fn seminumerical_vector_with(
    hessian: HessianEvaluator,
    who: &str,
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    step: f64,
    v: &[f64],
) -> Result<Matrix> {
    require_periodic(system, who)?;
    require_positive_step(step, who)?;
    let ndof = 3 * system.atoms.len();
    if v.len() != ndof {
        return Err(Gfn1Error::InvalidInput(format!(
            "{who}: direction length {} != 3*natoms {ndof}",
            v.len()
        )));
    }
    let mut k = Matrix::zeros(ndof, ndof);
    for (c, &vc) in v.iter().enumerate() {
        if vc == 0.0 {
            continue;
        }
        let slab = hessian_slab_with(hessian, system, params, options, pbc, c, step)?;
        for a in 0..ndof {
            for b in 0..ndof {
                k[(a, b)] += vc * slab[(a, b)];
            }
        }
    }
    Ok(k)
}

/// Shared implementation of the strain-mixed derivative `dH/d(ln V)`.
fn strain_hessian_derivative_with(
    hessian: HessianEvaluator,
    who: &str,
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    delta: f64,
) -> Result<Matrix> {
    require_periodic(system, who)?;
    if !(delta.is_finite() && delta > 0.0 && delta < 1.0) {
        return Err(Gfn1Error::InvalidInput(format!(
            "{who}: delta must be finite and in (0, 1) (got {delta})"
        )));
    }
    let ndof = 3 * system.atoms.len();
    let third = 1.0 / 3.0;
    let expanded = scale_lattice_isotropic(system, (1.0 + delta).powf(third))?;
    let compressed = scale_lattice_isotropic(system, (1.0 - delta).powf(third))?;
    let hp = hessian(&expanded, params, options, pbc)?.hessian;
    let hm = hessian(&compressed, params, options, pbc)?.hessian;
    let dln_v = ((1.0 + delta) / (1.0 - delta)).ln();
    let mut out = Matrix::zeros(ndof, ndof);
    for a in 0..ndof {
        for b in 0..ndof {
            out[(a, b)] = (hp[(a, b)] - hm[(a, b)]) / dln_v;
        }
    }
    Ok(out)
}

/// **Dense output.** The seminumerical periodic third derivative as `3N` slabs,
/// `slabs[c][(a, b)] = d(H_ab)/dR_c ~= d^3 E / dR_a dR_b dR_c` (Hartree/Bohr^3),
/// each column obtained by a central finite difference of the analytic
/// Gamma-point PBC Hessian along the atomic Cartesian DOF `c`.
///
/// Cost: `2 * 3N` analytic periodic Hessians (each of which re-runs the periodic
/// SCC from scratch).
///
/// Every slab is `(a, b)`-symmetric because the analytic Hessian is symmetrised,
/// but the tensor is **not** re-symmetrised across `c`: `slabs[c][(a,b)]` and
/// `slabs[a][(b,c)]` agree only to the finite-difference truncation order. That
/// is deliberate — it keeps invariance checks (e.g. the acoustic sum rule, which
/// contracts *different* slabs against each other) genuine tests rather than
/// artefacts of a symmetrisation step.
pub fn pbc_third_derivative_seminumerical_dense(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    step: f64,
) -> Result<Vec<Matrix>> {
    seminumerical_dense_with(
        pbc_gamma_hessian,
        "pbc_third_derivative_seminumerical_dense",
        system,
        params,
        options,
        pbc,
        step,
    )
}

/// **Vector output (memory-lean).** The directional third derivative
/// `K_ab = sum_c v_c d(H_ab)/dR_c`, a single `3N x 3N` matrix.
///
/// This is an **exact contraction** of the same per-DOF central differences the
/// dense mode builds, accumulated in the same index order, so it is bit-for-bit
/// equal to contracting [`pbc_third_derivative_seminumerical_dense`] with `v`
/// — not merely equal to finite-difference truncation order (which is what a
/// single displacement *along* `v` would give). DOFs with `v_c == 0.0` are
/// skipped, so the cost is `2 * nnz(v)` analytic periodic Hessians: for a
/// rigid-translation direction that is a third of the dense cost, and the peak
/// memory is one matrix instead of `3N`.
pub fn pbc_third_derivative_seminumerical_vector(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    step: f64,
    v: &[f64],
) -> Result<Matrix> {
    seminumerical_vector_with(
        pbc_gamma_hessian,
        "pbc_third_derivative_seminumerical_vector",
        system,
        params,
        options,
        pbc,
        step,
        v,
    )
}

/// **Dense output, Brillouin-zone sampled.** The seminumerical periodic third
/// derivative as `3N` slabs, `slabs[c][(a, b)] = d(H_ab)/dR_c`
/// (Hartree/Bohr^3), each column a central finite difference of the analytic
/// **k-point** PBC Hessian ([`pbc_kpoint_hessian`]) along the atomic Cartesian
/// DOF `c`. The mesh comes from `pbc.kmesh`, exactly as the k-point Hessian takes
/// it; a `1 x 1 x 1` mesh reduces this to
/// [`pbc_third_derivative_seminumerical_dense`].
///
/// Everything the Gamma version documents applies unchanged: absolute Cartesian
/// displacements with the lattice fixed, no re-symmetrisation across `c`, and the
/// finite-temperature caveat in the module docs (gated at `T = 0`, integer
/// occupations, gapped).
///
/// Cost: `2 * 3N` analytic k-point Hessians, each re-running the periodic SCC and
/// the complex CPXTB over the whole mesh.
///
/// This is the intended **verification base for a future analytic k-point FC3**:
/// it differentiates an already FD-gated analytic k-point Hessian, so it isolates
/// the one derivative the analytic route would have to add.
pub fn pbc_kpoint_third_derivative_seminumerical_dense(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    step: f64,
) -> Result<Vec<Matrix>> {
    seminumerical_dense_with(
        pbc_kpoint_hessian,
        "pbc_kpoint_third_derivative_seminumerical_dense",
        system,
        params,
        options,
        pbc,
        step,
    )
}

/// **Vector output (memory-lean), Brillouin-zone sampled.** The directional third
/// derivative `K_ab = sum_c v_c d(H_ab)/dR_c` from the analytic k-point Hessian.
///
/// As in the Gamma case this is an **exact contraction** of the same per-DOF
/// central differences the dense mode builds, accumulated in the same index
/// order, so it is bit-for-bit equal to contracting
/// [`pbc_kpoint_third_derivative_seminumerical_dense`] with `v`. DOFs with
/// `v_c == 0.0` are skipped, so the cost is `2 * nnz(v)` analytic k-point
/// Hessians — which is the affordable way to study **k-mesh convergence** of the
/// cubic force constants.
pub fn pbc_kpoint_third_derivative_seminumerical_vector(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    step: f64,
    v: &[f64],
) -> Result<Matrix> {
    seminumerical_vector_with(
        pbc_kpoint_hessian,
        "pbc_kpoint_third_derivative_seminumerical_vector",
        system,
        params,
        options,
        pbc,
        step,
        v,
    )
}

/// Scale a periodic system isotropically so that `V -> scale^3 * V`, keeping the
/// **fractional** coordinates of every atom fixed (frozen-ion convention).
///
/// Under an isotropic scaling `cell' = s * cell`, frozen fractional coordinates
/// mean `r' = cell' * f = s * cell * f = s * r`, so the Cartesian positions are
/// simply rescaled — no inverse-cell round trip is needed (and none is taken, to
/// keep the two displaced geometries exactly symmetric about the reference).
pub fn scale_lattice_isotropic(system: &PeriodicSystem, scale: f64) -> Result<PeriodicSystem> {
    if !(scale.is_finite() && scale > 0.0) {
        return Err(Gfn1Error::InvalidInput(format!(
            "scale_lattice_isotropic: scale must be finite and positive (got {scale})"
        )));
    }
    let lattice = require_periodic(system, "scale_lattice_isotropic")?;
    let cell = Mat3::from_columns(
        lattice.cell.col[0] * scale,
        lattice.cell.col[1] * scale,
        lattice.cell.col[2] * scale,
    );
    let mut out = system.clone();
    out.lattice = Some(Lattice::new(cell, lattice.periodic)?);
    for atom in &mut out.atoms {
        atom.position = atom.position * scale;
    }
    Ok(out)
}

/// **Strain-mixed third derivative**: `d(H_ab)/d(ln V)` (Hartree/Bohr^2, since
/// `ln V` is dimensionless), by a central difference of the analytic Gamma-point
/// PBC Hessian under *isotropic frozen-ion* volumetric strain.
///
/// The two displaced cells are `V(1 +/- delta)`, reached by scaling all three
/// lattice vectors by `(1 +/- delta)^(1/3)` with the atoms' fractional
/// coordinates frozen (see [`scale_lattice_isotropic`]). The denominator is the
/// **exact** log-volume separation
/// `Delta ln V = ln((1 + delta) / (1 - delta))`, not the leading-order `2 delta`;
/// with the exact denominator the estimator is a genuine central difference in
/// `ln V` and is `O(delta^2)`-accurate, which is what makes the `delta` vs
/// `delta/2` Richardson check in the gates meaningful.
///
/// Contracting this matrix with a mass-weighted normal mode gives that mode's
/// Grueneisen parameter directly; [`crate::pbc::gruneisen`] instead re-diagonalises
/// at both volumes, which additionally resolves mode crossings.
///
/// Cost: exactly `2` analytic periodic Hessians.
pub fn pbc_strain_hessian_derivative(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    delta: f64,
) -> Result<Matrix> {
    strain_hessian_derivative_with(
        pbc_gamma_hessian,
        "pbc_strain_hessian_derivative",
        system,
        params,
        options,
        pbc,
        delta,
    )
}

/// **Strain-mixed third derivative, Brillouin-zone sampled**: `d(H_ab)/d(ln V)`
/// (Hartree/Bohr^2) from the analytic **k-point** Hessian under isotropic
/// frozen-ion volumetric strain.
///
/// Identical in every convention to [`pbc_strain_hessian_derivative`] — the two
/// displaced cells are `V(1 +/- delta)` reached by scaling all three lattice
/// vectors by `(1 +/- delta)^(1/3)` with frozen fractional coordinates, and the
/// denominator is the exact log-volume separation `ln((1 + delta)/(1 - delta))` —
/// but each of the two Hessians is summed over `pbc.kmesh`. A `1 x 1 x 1` mesh
/// reduces it to the Gamma version.
///
/// The **k-mesh is held fixed** across the strain: the Monkhorst-Pack grid is
/// defined in fractional reciprocal coordinates, so scaling the cell scales the
/// sampled `k` vectors with the reciprocal lattice and the two volumes are
/// sampled at the same fractional points. That is the intended (and the only
/// self-consistent) convention here, and it is why the isotropic case needs no
/// mesh re-selection.
///
/// Cost: exactly `2` analytic k-point Hessians.
pub fn pbc_kpoint_strain_hessian_derivative(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    options: &ElectronicOptions,
    pbc: &PbcOptions,
    delta: f64,
) -> Result<Matrix> {
    strain_hessian_derivative_with(
        pbc_kpoint_hessian,
        "pbc_kpoint_strain_hessian_derivative",
        system,
        params,
        options,
        pbc,
        delta,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diamond_primitive() -> PeriodicSystem {
        // 2-atom primitive fcc cell of diamond, a = 3.567 A.
        PeriodicSystem::from_xyz_str(
            "2\nLattice=\"0.0 1.7835 1.7835 1.7835 0.0 1.7835 1.7835 1.7835 0.0\" pbc=\"T T T\"\n\
             C 0.000000 0.000000 0.000000\n\
             C 0.891750 0.891750 0.891750\n",
            0.0,
            false,
        )
        .unwrap()
    }

    // Isotropic frozen-ion scaling must change the volume by exactly `s^3` and
    // leave every fractional coordinate untouched.
    #[test]
    fn isotropic_scaling_preserves_fractional_coordinates() {
        let base = diamond_primitive();
        let s = 1.01_f64;
        let scaled = scale_lattice_isotropic(&base, s).unwrap();
        let v0 = base.lattice.unwrap().volume();
        let v1 = scaled.lattice.unwrap().volume();
        assert!(
            ((v1 / v0) - s.powi(3)).abs() < 1.0e-12,
            "volume ratio {} vs s^3 {}",
            v1 / v0,
            s.powi(3)
        );
        for (a, b) in base.atoms.iter().zip(scaled.atoms.iter()) {
            let fa = base.lattice.unwrap().frac_of(a.position);
            let fb = scaled.lattice.unwrap().frac_of(b.position);
            assert!(
                (fa - fb).norm() < 1.0e-12,
                "fractional coordinate drifted under isotropic scaling"
            );
        }
    }

    // A non-periodic system must be rejected rather than silently panicking on
    // the `unwrap` inside the periodic Hessian.
    #[test]
    fn non_periodic_input_is_rejected() {
        let molecule =
            PeriodicSystem::from_xyz_str("1\nbare C\nC 0.0 0.0 0.0\n", 0.0, false).unwrap();
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        let err = pbc_third_derivative_seminumerical_dense(
            &molecule,
            &params,
            &ElectronicOptions::default(),
            &PbcOptions::default(),
            1.0e-3,
        );
        assert!(err.is_err(), "a non-periodic system must be rejected");
        let err = pbc_strain_hessian_derivative(
            &molecule,
            &params,
            &ElectronicOptions::default(),
            &PbcOptions::default(),
            5.0e-3,
        );
        assert!(err.is_err(), "a non-periodic system must be rejected");
        // The k-point family must guard identically (it shares the guards, but
        // a re-wiring mistake would be silent otherwise).
        let err = pbc_kpoint_third_derivative_seminumerical_dense(
            &molecule,
            &params,
            &ElectronicOptions::default(),
            &PbcOptions::default(),
            1.0e-3,
        );
        assert!(err.is_err(), "a non-periodic system must be rejected");
        let err = pbc_kpoint_strain_hessian_derivative(
            &molecule,
            &params,
            &ElectronicOptions::default(),
            &PbcOptions::default(),
            5.0e-3,
        );
        assert!(err.is_err(), "a non-periodic system must be rejected");
    }

    // Bad finite-difference controls must error, not produce inf/NaN slabs.
    #[test]
    fn invalid_step_and_delta_are_rejected() {
        let base = diamond_primitive();
        let params = Gfn1Parameters::resolve(None).expect("GFN1 parameter resolution failed");
        assert!(pbc_third_derivative_seminumerical_dense(
            &base,
            &params,
            &ElectronicOptions::default(),
            &PbcOptions::default(),
            0.0,
        )
        .is_err());
        assert!(pbc_strain_hessian_derivative(
            &base,
            &params,
            &ElectronicOptions::default(),
            &PbcOptions::default(),
            1.5,
        )
        .is_err());
        let v = vec![1.0; 5];
        assert!(pbc_third_derivative_seminumerical_vector(
            &base,
            &params,
            &ElectronicOptions::default(),
            &PbcOptions::default(),
            1.0e-3,
            &v,
        )
        .is_err());
        assert!(pbc_kpoint_third_derivative_seminumerical_dense(
            &base,
            &params,
            &ElectronicOptions::default(),
            &PbcOptions::default(),
            -1.0e-3,
        )
        .is_err());
        assert!(pbc_kpoint_third_derivative_seminumerical_vector(
            &base,
            &params,
            &ElectronicOptions::default(),
            &PbcOptions::default(),
            1.0e-3,
            &v,
        )
        .is_err());
        assert!(pbc_kpoint_strain_hessian_derivative(
            &base,
            &params,
            &ElectronicOptions::default(),
            &PbcOptions::default(),
            0.0,
        )
        .is_err());
    }
}
