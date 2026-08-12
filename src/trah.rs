//! Trust-Region Augmented Hessian (TRAH) second-order SCF — matrix-free foundation.
//!
//! A robust fallback for the experimental exchange-augmented SCC (MFX/OFX). The off-diagonal,
//! density-matrix-dependent exchange Fock can defeat charge-vector mixing (and even commutator DIIS
//! far from the solution); TRAH instead minimises the electronic energy **directly over orbital
//! rotations** `C → C·exp(κ)` with `κ` real and antisymmetric in the MO indices, restricted to the
//! occupied–virtual block (occ–occ / virt–virt rotations are redundant). The Newton step uses a
//! **matrix-free** orbital Hessian: the Hessian is never assembled — only Hessian–vector products
//! `H v` are formed from the linear Fock response `δF[δP]`. A trust region / augmented-Hessian
//! level shift globalises the step so it is robust far from convergence.
//!
//! This module is *pure orbital-rotation algebra*. It takes the current MOs, occupations, the
//! MO-basis Fock `F^MO = CᵀFC`, and a closure `fock_response: δP ↦ δF` (the linear change of the AO
//! Fock under a density change — for the SCC this is the second-order charge kernel + MFX + OFX
//! kernels, each already linear in `P`). It is therefore independent of which energy functional is
//! used, and is validated against both an analytic model functional and (via the SCC wiring) the
//! real exchange functional — every derivative is finite-difference-gated.
//!
//! ## Formulae (closed/fractional-occupation restricted)
//! For a rotation `κ` whose only free parameters are the occ–virt entries `κ_{ai}` (`κ_{ia}=−κ_{ai}`):
//! - density response `δP = Σ_{ai} κ_{ai} Δn_{ai} (c_a c_iᵀ + c_i c_aᵀ)`, `Δn_{ai}=n_i−n_a>0`;
//! - orbital gradient `g_{ai} = 2 Δn_{ai} F^MO_{ai}`;
//! - Hessian–vector `H v|_{ai} = 2 Δn_{ai} ( [F^MO, K_v] + Cᵀ δF[δP(v)] C )_{ai}`, where `K_v` is the
//!   full antisymmetric rotation matrix of `v`. (At a canonical point `F^MO=diag(ε)` the commutator
//!   reduces to the familiar `(ε_a−ε_i)v_{ai}`; the full commutator keeps it exact at any `C`.)
//!
//! ## References (no brand names)
//! Primary: **B. Helmich-Paris, "A trust-region augmented Hessian implementation for restricted and
//! unrestricted Hartree–Fock and Kohn–Sham methods", *J. Chem. Phys.* 154, 164104 (2021)** — the
//! TRAH-SCF method this module follows: minimise `E(κ)=⟨0̃|H|0̃⟩`, `|0̃⟩=exp(κ̂)|0⟩`, via a Newton step
//! `Hκ=−g` whose orbital Hessian is applied **matrix-free** through a Fock-matrix linear
//! transformation (avoiding the `O(N⁵)` AO→MO transform), globalised by a trust region `‖κ‖₂≤h`;
//! TRAH converges even when DIIS diverges (its focus is hard cases — magnetically-coupled /
//! broken-symmetry). Foundations: Helgaker, Jørgensen & Olsen, *Molecular Electronic-Structure
//! Theory* (Wiley, 2000), ch. 10 (orbital rotations, exponential parametrisation, second-order SCF);
//! Fletcher, *Practical Methods of Optimization* (Wiley, 1987) (augmented Hessian / trust region);
//! E. R. Davidson, *J. Comput. Phys.* 17, 87 (1975) (iterative diagonalisation). Off by default;
//! experimental.
//!
//! NB this implementation solves the trust-region Newton step `(H+λI)κ=−g` by **conjugate gradients**
//! with a level-shift line search (the `λ` is the augmented-Hessian shift), rather than the paper's
//! Davidson diagonalisation of the augmented Hessian; both target the same regularised step. It uses
//! fixed (integer/closed-shell) occupations — the orbital-rotation manifold — so it is appropriate for
//! gapped systems, not fractional-occupation metals.

use crate::error::Result;
use crate::linalg::Matrix;

/// The occupied–virtual orbital-rotation space: one parameter per `(i,a)` with `n_i > n_a`.
#[derive(Clone, Debug)]
pub struct OrbitalRotationSpace {
    /// `(i, a, Δn=n_i−n_a)` for every rotation parameter (`i` "more occupied", `a` "less occupied").
    pub pairs: Vec<(usize, usize, f64)>,
    /// Number of molecular orbitals.
    pub nmo: usize,
}

impl OrbitalRotationSpace {
    /// Build from occupation numbers: a parameter for each ordered pair with a positive occupation
    /// difference (so a fractional/Fermi-smeared occupation still yields a well-posed, full-rank
    /// rotation space — pairs with equal occupation carry no energy gradient and are dropped).
    pub fn from_occupations(occ: &[f64]) -> Self {
        let nmo = occ.len();
        let mut pairs = Vec::new();
        for i in 0..nmo {
            for a in 0..nmo {
                let dn = occ[i] - occ[a];
                if dn > 1.0e-10 {
                    pairs.push((i, a, dn));
                }
            }
        }
        Self { pairs, nmo }
    }

    /// Number of rotation parameters.
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Whether the rotation space is empty (no occ–virt gradient directions).
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// The full antisymmetric MO rotation matrix `K_v` of a rotation vector `v`
    /// (`K[a,i]=v`, `K[i,a]=−v`).
    pub fn rotation_matrix(&self, v: &[f64]) -> Matrix {
        let mut k = Matrix::zeros(self.nmo, self.nmo);
        for (p, &(i, a, _dn)) in self.pairs.iter().enumerate() {
            k[(a, i)] += v[p];
            k[(i, a)] -= v[p];
        }
        k
    }
}

/// Matrix exponential `exp(K)` of a real square matrix via scaling-and-squaring with a 12-term
/// Taylor series. For an antisymmetric `K` the result is orthogonal to machine precision, so
/// `C·exp(K)` preserves `CᵀSC = I` exactly (the rotation stays on the Stiefel manifold).
pub fn expm(k: &Matrix) -> Result<Matrix> {
    let n = k.rows();
    if n == 0 {
        return Ok(Matrix::zeros(0, 0));
    }
    // Scale so ‖K/2^s‖ is small, then square s times.
    let norm = k.as_slice().iter().map(|x| x.abs()).fold(0.0, f64::max) * (n as f64);
    let theta = 0.5_f64;
    let s = if norm > theta {
        (norm / theta).log2().ceil().max(0.0) as i32
    } else {
        0
    };
    let scale = 2.0_f64.powi(s);
    let mut scaled = k.clone();
    if s > 0 {
        for x in scaled.as_mut_slice() {
            *x /= scale;
        }
    }
    // acc = I + A + A²/2! + … + A¹²/12!
    let mut acc = Matrix::identity(n);
    let mut term = Matrix::identity(n);
    for kk in 1..=12usize {
        term = term.matmul(&scaled)?;
        let inv = 1.0 / (kk as f64);
        for x in term.as_mut_slice() {
            *x *= inv;
        }
        for (a, t) in acc.as_mut_slice().iter_mut().zip(term.as_slice()) {
            *a += *t;
        }
    }
    for _ in 0..s {
        acc = acc.matmul(&acc)?;
    }
    Ok(acc)
}

/// Rotate the MOs by `v`: `C' = C · exp(K_v)` (AO×nmo · nmo×nmo). `C'` stays `S`-orthonormal.
pub fn rotate_mos(mo: &Matrix, space: &OrbitalRotationSpace, v: &[f64]) -> Result<Matrix> {
    let q = expm(&space.rotation_matrix(v))?;
    mo.matmul(&q)
}

/// Density matrix `P_{μν} = Σ_p n_p C_{μp} C_{νp}` from MOs and occupations.
pub fn density_from_mos(mo: &Matrix, occ: &[f64]) -> Matrix {
    let n = mo.rows();
    let nmo = mo.cols();
    let mut p = Matrix::zeros(n, n);
    for r in 0..nmo {
        let nr = occ[r];
        if nr.abs() < 1.0e-14 {
            continue;
        }
        for mu in 0..n {
            let c = nr * mo[(mu, r)];
            if c == 0.0 {
                continue;
            }
            for nu in 0..n {
                p[(mu, nu)] += c * mo[(nu, r)];
            }
        }
    }
    p
}

/// Project an AO matrix into the MO basis: `M^MO = Cᵀ M C`.
pub fn to_mo_basis(mo: &Matrix, ao: &Matrix) -> Result<Matrix> {
    mo.transpose().matmul(ao)?.matmul(mo)
}

/// Orbital gradient `g_{ai} = 2 Δn_{ai} F^MO_{ai}`, one entry per rotation parameter.
/// `fock_mo = CᵀFC` for the AO Fock `F = ∂E/∂P` built at the current density.
pub fn orbital_gradient(fock_mo: &Matrix, space: &OrbitalRotationSpace) -> Vec<f64> {
    space
        .pairs
        .iter()
        .map(|&(i, a, dn)| 2.0 * dn * fock_mo[(a, i)])
        .collect()
}

/// Jacobi preconditioner for the inner Newton CG: the magnitude of the orbital-Hessian diagonal
/// estimate `|2 Δn_ai (ε_a − ε_i)|` (ε from the MO-Fock diagonal). `|·|` keeps it positive through
/// level crossings / saddles (the λ-shift handles the actual indefiniteness), and a **relative**
/// floor `≥ 1e-2·max` caps its condition number at ~100 — capturing the gap spread that slows an
/// unpreconditioned CG without over-amplifying near-degenerate directions.
pub fn jacobi_preconditioner(fock_mo: &Matrix, space: &OrbitalRotationSpace) -> Vec<f64> {
    let raw: Vec<f64> = space
        .pairs
        .iter()
        .map(|&(i, a, dn)| (2.0 * dn * (fock_mo[(a, a)] - fock_mo[(i, i)])).abs())
        .collect();
    let maxd = raw.iter().cloned().fold(0.0_f64, f64::max).max(1.0e-12);
    let floor = 1.0e-2 * maxd;
    raw.iter().map(|&d| d.max(floor)).collect()
}

/// Density response `δP[v]` (AO) to a rotation `v`:
/// `δP_{μν} = Σ_{ai} v_{ai} Δn_{ai} (C_{μa}C_{νi} + C_{μi}C_{νa})`.
pub fn density_response(mo: &Matrix, space: &OrbitalRotationSpace, v: &[f64]) -> Matrix {
    // Small systems: the explicit rank update (bit-reproducible — the TRAH model gate pins it to
    // 1e-8). Large systems: the algebraically-identical `δP = C·M·Cᵀ` GEMM, which is O(N³) instead
    // of the O(N⁴) pair×N² loop — the dominant cost of each Hessian–vector product (the TRAH inner
    // CG calls this once per product). The two agree to round-off (`density_response_*_agree`).
    if mo.rows() < 96 {
        density_response_explicit(mo, space, v)
    } else {
        density_response_gemm(mo, space, v)
    }
}

fn density_response_explicit(mo: &Matrix, space: &OrbitalRotationSpace, v: &[f64]) -> Matrix {
    let n = mo.rows();
    let mut dp = Matrix::zeros(n, n);
    for (p, &(i, a, dn)) in space.pairs.iter().enumerate() {
        let c = v[p] * dn;
        if c == 0.0 {
            continue;
        }
        for mu in 0..n {
            let ca = c * mo[(mu, a)];
            let ci = c * mo[(mu, i)];
            for nu in 0..n {
                dp[(mu, nu)] += ca * mo[(nu, i)] + ci * mo[(nu, a)];
            }
        }
    }
    dp
}

fn density_response_gemm(mo: &Matrix, space: &OrbitalRotationSpace, v: &[f64]) -> Matrix {
    // `M_ai = M_ia = v_p Δn_ai` (symmetric MO-basis response); δP = C·M·Cᵀ.
    let nmo = mo.cols();
    let mut m = Matrix::zeros(nmo, nmo);
    for (p, &(i, a, dn)) in space.pairs.iter().enumerate() {
        let val = v[p] * dn;
        m[(a, i)] = val;
        m[(i, a)] = val;
    }
    let cm = mo.matmul(&m).expect("C·M dimensions match by construction");
    cm.matmul(&mo.transpose())
        .expect("(C·M)·Cᵀ dimensions match by construction")
}

/// Matrix-free orbital Hessian–vector product `H v`. `fock_mo = CᵀFC` (current MO Fock);
/// `fock_response(δP) → δF` is the **linear** change of the AO Fock under a density change (the
/// second-order SCC charge kernel + MFX + OFX, all already linear in `P`). Returns one entry per
/// rotation parameter:
/// `Hv_{ai} = 2 Δn_{ai} ( [F^MO, K_v]_{ai} + (Cᵀ δF[δP(v)] C)_{ai} )`.
pub fn hessian_vector<F>(
    v: &[f64],
    space: &OrbitalRotationSpace,
    mo: &Matrix,
    fock_mo: &Matrix,
    fock_response: &F,
) -> Result<Vec<f64>>
where
    F: Fn(&Matrix) -> Matrix,
{
    // Commutator term [F^MO, K_v] = F^MO·K − K·F^MO.
    let kmat = space.rotation_matrix(v);
    let fk = fock_mo.matmul(&kmat)?;
    let kf = kmat.matmul(fock_mo)?;
    // Response term Cᵀ δF[δP(v)] C.
    let dp = density_response(mo, space, v);
    let df = fock_response(&dp);
    let df_mo = to_mo_basis(mo, &df)?;
    let mut hv = vec![0.0; space.len()];
    for (p, &(i, a, dn)) in space.pairs.iter().enumerate() {
        let comm = fk[(a, i)] - kf[(a, i)];
        hv[p] = 2.0 * dn * (comm + df_mo[(a, i)]);
    }
    Ok(hv)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// Result of one trust-region augmented-Hessian Newton step.
#[derive(Clone, Debug)]
pub struct TrahStep {
    /// The rotation vector `κ` (one entry per rotation parameter).
    pub kappa: Vec<f64>,
    /// `‖κ‖₂`.
    pub step_norm: f64,
    /// Predicted energy change `gᵀκ + ½ κᵀHκ`.
    pub predicted_reduction: f64,
    /// Level shift `λ` applied (`(H+λI)κ=−g`); `0` if the pure Newton step was inside the trust radius.
    pub level_shift: f64,
    /// Davidson micro-iterations used.
    pub iterations: usize,
}

/// Solve the level-shifted Newton equations `(H + λI) κ = −g` matrix-free by conjugate gradients,
/// choosing the smallest `λ ≥ 0` (augmented-Hessian regularisation) such that `‖κ‖ ≤ trust_radius`.
/// Only `hv` (a Hessian–vector product) is used — the Hessian is never assembled. This is the inner
/// TRAH solver; the `λ` search makes it robust when `H` is indefinite (far from convergence).
pub fn trust_region_newton_step<H>(
    g: &[f64],
    precond: &[f64],
    hv: &H,
    trust_radius: f64,
    cg_tol: f64,
    max_cg: usize,
) -> Result<TrahStep>
where
    H: Fn(&[f64]) -> Result<Vec<f64>>,
{
    // `precond` is the Jacobi diagonal used **only** to seed the very first (λ=0) Newton attempt,
    // which is the hot path once near convergence (H positive-definite, the λ-search is skipped).
    // The λ-search itself (indefinite H, far from convergence) runs *unpreconditioned* CG, whose
    // negative-curvature truncation the trust-region logic is tuned for — preconditioning there
    // changes the truncated step and destabilises the bracket/bisection.
    let n = g.len();
    if n == 0 {
        return Ok(TrahStep {
            kappa: Vec::new(),
            step_norm: 0.0,
            predicted_reduction: 0.0,
            level_shift: 0.0,
            iterations: 0,
        });
    }
    // Solve (H+λI)κ = −g by **Jacobi-preconditioned** CG for a given λ. The preconditioner
    // `M = diag(H)+λ ≈ 2Δn_ai(ε_a−ε_i)+λ` (`precond`, floored positive) absorbs the orbital-gap
    // spread, so PCG converges in O(10) products instead of O(N) — the dominant TRAH cost on a
    // near-degenerate system where each Hv is an O(N³) Fock response.
    let solve = |lambda: f64, use_precond: bool| -> Result<(Vec<f64>, usize)> {
        let mut x = vec![0.0; n];
        // r = −g − (H+λI)x = −g at x=0.
        let mut r: Vec<f64> = g.iter().map(|v| -v).collect();
        let minv: Vec<f64> = if use_precond {
            precond
                .iter()
                .map(|&d| 1.0 / (d + lambda).max(1.0e-6))
                .collect()
        } else {
            vec![1.0; n] // unpreconditioned CG (≡ M = I) for the indefinite λ-search
        };
        let mut z: Vec<f64> = r.iter().zip(minv.iter()).map(|(ri, mi)| ri * mi).collect();
        let mut p = z.clone();
        let mut rz = dot(&r, &z);
        let mut iters = 0;
        let gnorm = norm2(g).max(1.0e-30);
        for _ in 0..max_cg {
            iters += 1;
            let mut hp = hv(&p)?;
            for (h, pv) in hp.iter_mut().zip(p.iter()) {
                *h += lambda * pv;
            }
            let denom = dot(&p, &hp);
            // Negative/zero curvature ⇒ `H+λI` is not PD along `p`; stop and let the λ-search raise
            // the shift (the augmented-Hessian regularisation) until the system is PD.
            if denom <= 1.0e-30 {
                break;
            }
            let alpha = rz / denom;
            for (xi, pi) in x.iter_mut().zip(p.iter()) {
                *xi += alpha * pi;
            }
            for (ri, hpi) in r.iter_mut().zip(hp.iter()) {
                *ri -= alpha * hpi;
            }
            if norm2(&r) < cg_tol * gnorm {
                break;
            }
            for (zi, (ri, mi)) in z.iter_mut().zip(r.iter().zip(minv.iter())) {
                *zi = ri * mi;
            }
            let rz_new = dot(&r, &z);
            let beta = rz_new / rz;
            for (pi, zi) in p.iter_mut().zip(z.iter()) {
                *pi = zi + beta * *pi;
            }
            rz = rz_new;
        }
        Ok((x, iters))
    };

    // Find the smallest λ ≥ 0 keeping ‖κ‖ ≤ trust_radius. ‖κ(λ)‖ decreases monotonically in λ.
    // (Preconditioning is plumbed but left off: it changes the negative-curvature-truncated step in
    // the indefinite regime, and the O(N³) `density_response` + early ADIIS→TRAH hand-off already
    // make the continuation fast without it.)
    let (mut kappa, mut iters) = solve(0.0, false)?;
    let mut level_shift = 0.0;
    let mut total_iters = iters;
    if norm2(&kappa) > trust_radius {
        // Bracket then bisect λ. Start from a scale set by the gradient / trust radius.
        let mut lo = 0.0_f64;
        let mut hi = (norm2(g) / trust_radius).max(1.0e-6);
        // Grow hi until the step fits.
        for _ in 0..60 {
            let (k, it) = solve(hi, false)?;
            total_iters += it;
            if norm2(&k) <= trust_radius {
                kappa = k;
                iters = it;
                break;
            }
            lo = hi;
            hi *= 2.0;
        }
        // Bisection to land near the trust-region boundary.
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            let (k, it) = solve(mid, false)?;
            total_iters += it;
            let kn = norm2(&k);
            if kn > trust_radius {
                lo = mid;
            } else {
                hi = mid;
                kappa = k;
                iters = it;
            }
            if (hi - lo) <= 1.0e-3 * hi {
                break;
            }
        }
        level_shift = hi;
    }

    // Robustness: if the (possibly indefinite) solve degenerated to a near-zero step while the
    // gradient is nonzero, fall back to a steepest-descent step scaled to the trust radius — always a
    // descent direction, so the macro-iteration still makes progress.
    let gnorm = norm2(g);
    if norm2(&kappa) < 1.0e-12 && gnorm > 1.0e-12 {
        let scale = trust_radius.min(gnorm) / gnorm;
        kappa = g.iter().map(|gi| -scale * gi).collect();
    }

    // Predicted reduction gᵀκ + ½κᵀHκ.
    let hk = hv(&kappa)?;
    let predicted_reduction = dot(g, &kappa) + 0.5 * dot(&kappa, &hk);
    let step_norm = norm2(&kappa);
    let _ = iters;
    Ok(TrahStep {
        kappa,
        step_norm,
        predicted_reduction,
        level_shift,
        iterations: total_iters,
    })
}

/// Options for the TRAH macro-iteration ([`run_trah_scf`]).
#[derive(Clone, Copy, Debug)]
pub struct TrahOptions {
    /// Maximum macro-iterations.
    pub max_iter: usize,
    /// Convergence threshold on the orbital-gradient norm `‖g‖₂`.
    pub grad_tol: f64,
    /// Initial trust radius (max `‖κ‖` per step).
    pub trust_radius: f64,
    /// Hard cap on the (adaptively grown) trust radius.
    pub max_trust: f64,
    /// Relative CG tolerance for the inner level-shifted Newton solve.
    pub cg_tol: f64,
    /// Maximum CG micro-iterations per inner solve.
    pub max_cg: usize,
}

impl Default for TrahOptions {
    fn default() -> Self {
        Self {
            max_iter: 100,
            grad_tol: 1.0e-6,
            trust_radius: 0.4,
            max_trust: 1.0,
            cg_tol: 1.0e-8,
            max_cg: 200,
        }
    }
}

/// Result of a TRAH macro-iteration.
#[derive(Clone, Debug)]
pub struct TrahScfResult {
    /// Converged (or last) MOs, `S`-orthonormal (AO×nmo).
    pub mo: Matrix,
    /// Density matrix at `mo`.
    pub density: Matrix,
    /// Total energy at `mo` (whatever `fock_energy` returns).
    pub energy: f64,
    /// Final orbital-gradient norm.
    pub gradient_norm: f64,
    /// Macro-iterations performed.
    pub iterations: usize,
    /// Whether `‖g‖ < grad_tol` was reached.
    pub converged: bool,
}

/// Run the **Trust-Region Augmented Hessian** SCF to convergence by direct second-order
/// energy minimisation over orbital rotations. `initial_mo` are `S`-orthonormal MOs (e.g. from the
/// core-Hamiltonian guess), `occ` the (fixed) occupation numbers; `fock_energy(P) → (F, E)` builds
/// the AO Fock `F = ∂E/∂P` and the energy at a density, and `fock_response(δP) → δF` is the **linear**
/// Fock response (second-order SCC charge kernel + MFX + OFX). Each macro-iteration takes a
/// trust-region augmented-Hessian Newton step (matrix-free), with the trust radius adapted from the
/// ratio of actual to predicted energy reduction (standard Fletcher update). Robust where DIIS on the
/// off-diagonal exchange Fock stalls. Non-PBC; fixed occupations (gapped/closed-shell).
pub fn run_trah_scf<FE, FR>(
    initial_mo: &Matrix,
    occ: &[f64],
    fock_energy: FE,
    fock_response: FR,
    options: &TrahOptions,
) -> Result<TrahScfResult>
where
    FE: Fn(&Matrix) -> (Matrix, f64),
    FR: Fn(&Matrix) -> Matrix,
{
    let space = OrbitalRotationSpace::from_occupations(occ);
    let mut mo = initial_mo.clone();
    let mut density = density_from_mos(&mo, occ);
    let (mut fock, mut energy) = fock_energy(&density);
    let mut trust = options.trust_radius;
    let mut converged = false;
    let mut iterations = 0;
    let mut gnorm = f64::INFINITY;
    if space.is_empty() {
        // No occupied–virtual rotation freedom (e.g. fully occupied) — already stationary.
        return Ok(TrahScfResult {
            mo,
            density,
            energy,
            gradient_norm: 0.0,
            iterations: 0,
            converged: true,
        });
    }
    for it in 1..=options.max_iter {
        iterations = it;
        let fmo = to_mo_basis(&mo, &fock)?;
        let g = orbital_gradient(&fmo, &space);
        gnorm = norm2(&g);
        if gnorm < options.grad_tol {
            converged = true;
            break;
        }
        let resp = |dp: &Matrix| fock_response(dp);
        let hv = |v: &[f64]| hessian_vector(v, &space, &mo, &fmo, &resp);
        let precond = jacobi_preconditioner(&fmo, &space);
        let step =
            trust_region_newton_step(&g, &precond, &hv, trust, options.cg_tol, options.max_cg)?;
        // Trial step and trust-region ratio ρ = actual / predicted reduction.
        let trial_mo = rotate_mos(&mo, &space, &step.kappa)?;
        let trial_density = density_from_mos(&trial_mo, occ);
        let (trial_fock, trial_energy) = fock_energy(&trial_density);
        let actual = trial_energy - energy;
        let predicted = step.predicted_reduction;
        let rho = if predicted < -1.0e-14 {
            actual / predicted
        } else {
            // Non-descent model prediction: accept only a genuine energy decrease.
            if actual < 0.0 {
                1.0
            } else {
                -1.0
            }
        };
        if rho > 0.1 && actual < 0.0 {
            // Accept the step.
            mo = trial_mo;
            density = trial_density;
            fock = trial_fock;
            energy = trial_energy;
            if rho > 0.75 && step.step_norm > 0.8 * trust {
                trust = (2.0 * trust).min(options.max_trust);
            }
        } else {
            // Reject and shrink the trust region.
            trust *= 0.25;
            if trust < 1.0e-10 {
                break;
            }
        }
    }
    Ok(TrahScfResult {
        mo,
        density,
        energy,
        gradient_norm: gnorm,
        iterations,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A self-contained, exactly-differentiable model SCF functional
    /// `E[P] = Tr[hP] + ¼ Tr[P A P A]` (a stand-in for one-electron + a quartic two-electron term),
    /// in an orthonormal basis (`S=I`). Then `F = ∂E/∂P = h + ½ A P A` and the linear Fock response
    /// is `δF[δP] = ½ A δP A`. `h`, `A` symmetric ⇒ `F` symmetric.
    struct Model {
        h: Matrix,
        a: Matrix,
    }
    impl Model {
        fn fock(&self, p: &Matrix) -> Matrix {
            let apa = self.a.matmul(p).unwrap().matmul(&self.a).unwrap();
            let n = p.rows();
            let mut f = Matrix::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    f[(i, j)] = self.h[(i, j)] + 0.5 * apa[(i, j)];
                }
            }
            f
        }
        fn response(&self, dp: &Matrix) -> Matrix {
            let adpa = self.a.matmul(dp).unwrap().matmul(&self.a).unwrap();
            let n = dp.rows();
            let mut df = Matrix::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    df[(i, j)] = 0.5 * adpa[(i, j)];
                }
            }
            df
        }
        fn energy(&self, p: &Matrix) -> f64 {
            let n = p.rows();
            let mut e = 0.0;
            for i in 0..n {
                for j in 0..n {
                    e += self.h[(i, j)] * p[(i, j)];
                }
            }
            let apa = self.a.matmul(p).unwrap().matmul(&self.a).unwrap();
            let mut quad = 0.0;
            for i in 0..n {
                for j in 0..n {
                    quad += p[(i, j)] * apa[(i, j)];
                }
            }
            e + 0.25 * quad
        }
    }

    // Deterministic pseudo-random symmetric matrix.
    fn sym(n: usize, seed: u64) -> Matrix {
        let mut s = seed;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };
        let mut m = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                let v = next();
                m[(i, j)] = v;
                m[(j, i)] = v;
            }
        }
        m
    }

    // Orthonormal MO matrix: eigenvectors of a random symmetric matrix (S=I).
    fn orthonormal(n: usize, seed: u64) -> Matrix {
        crate::linalg::symmetric_eigen(&sym(n, seed))
            .unwrap()
            .vectors
    }

    fn model_and_state(n: usize, nocc: usize) -> (Model, Matrix, Vec<f64>) {
        let model = Model {
            h: sym(n, 1),
            a: sym(n, 7),
        };
        let mo = orthonormal(n, 19);
        let occ: Vec<f64> = (0..n).map(|i| if i < nocc { 2.0 } else { 0.0 }).collect();
        (model, mo, occ)
    }

    /// `exp(K)` of an antisymmetric `K` is orthogonal (`QᵀQ = I`).
    #[test]
    fn expm_of_skew_is_orthogonal() {
        let n = 5;
        let s = sym(n, 3);
        let mut k = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                k[(i, j)] = s[(i, j)] - s[(j, i)]; // antisymmetric part (×2, fine)
            }
        }
        let q = expm(&k).unwrap();
        let qtq = q.transpose().matmul(&q).unwrap();
        let mut maxoff = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let target = if i == j { 1.0 } else { 0.0 };
                maxoff = maxoff.max((qtq[(i, j)] - target).abs());
            }
        }
        assert!(maxoff < 1.0e-12, "exp(skew) not orthogonal: {maxoff:.3e}");
    }

    /// **M4.2 gate** — the analytic orbital gradient equals a central finite difference of the model
    /// energy `E(C·exp(κ))` along each rotation parameter.
    #[test]
    fn orbital_gradient_matches_fd() {
        let (model, mo, occ) = model_and_state(6, 3);
        let space = OrbitalRotationSpace::from_occupations(&occ);
        let p = density_from_mos(&mo, &occ);
        let f = model.fock(&p);
        let fmo = to_mo_basis(&mo, &f).unwrap();
        let g = orbital_gradient(&fmo, &space);
        let eps = 1.0e-5;
        let mut maxdiff = 0.0_f64;
        for p_idx in 0..space.len() {
            let mut vp = vec![0.0; space.len()];
            vp[p_idx] = eps;
            let mut vm = vec![0.0; space.len()];
            vm[p_idx] = -eps;
            let ep = model.energy(&density_from_mos(
                &rotate_mos(&mo, &space, &vp).unwrap(),
                &occ,
            ));
            let em = model.energy(&density_from_mos(
                &rotate_mos(&mo, &space, &vm).unwrap(),
                &occ,
            ));
            let fd = (ep - em) / (2.0 * eps);
            maxdiff = maxdiff.max((g[p_idx] - fd).abs());
        }
        assert!(maxdiff < 1.0e-6, "orbital gradient vs FD: {maxdiff:.3e}");
    }

    /// **M4.3 gate** — the matrix-free Hessian–vector product equals a central finite difference of
    /// the orbital gradient `g(t v)` (directional derivative), for a random direction `v`.
    #[test]
    fn hessian_vector_matches_fd() {
        let (model, mo, occ) = model_and_state(6, 3);
        let space = OrbitalRotationSpace::from_occupations(&occ);
        let p = density_from_mos(&mo, &occ);
        let f = model.fock(&p);
        let fmo = to_mo_basis(&mo, &f).unwrap();
        let resp = |dp: &Matrix| model.response(dp);
        // Random direction.
        let v: Vec<f64> = (0..space.len())
            .map(|i| 0.1 * ((i as f64 * 0.7).sin()))
            .collect();
        let hv = hessian_vector(&v, &space, &mo, &fmo, &resp).unwrap();
        // FD of the gradient along v: g(±εv) at the *rotated* orbitals (Fock rebuilt there).
        let grad_at = |scale: f64| -> Vec<f64> {
            let vv: Vec<f64> = v.iter().map(|x| x * scale).collect();
            let mor = rotate_mos(&mo, &space, &vv).unwrap();
            let pr = density_from_mos(&mor, &occ);
            let fr = model.fock(&pr);
            let fmor = to_mo_basis(&mor, &fr).unwrap();
            orbital_gradient(&fmor, &space)
        };
        let eps = 1.0e-5;
        let gp = grad_at(eps);
        let gm = grad_at(-eps);
        let mut maxdiff = 0.0_f64;
        for i in 0..space.len() {
            let fd = (gp[i] - gm[i]) / (2.0 * eps);
            maxdiff = maxdiff.max((hv[i] - fd).abs());
        }
        assert!(maxdiff < 1.0e-5, "Hessian-vector vs FD: {maxdiff:.3e}");
    }

    /// The orbital Hessian is symmetric: `uᵀ(Hv) = vᵀ(Hu)`.
    #[test]
    fn hessian_is_symmetric() {
        let (model, mo, occ) = model_and_state(6, 3);
        let space = OrbitalRotationSpace::from_occupations(&occ);
        let p = density_from_mos(&mo, &occ);
        let fmo = to_mo_basis(&mo, &model.fock(&p)).unwrap();
        let resp = |dp: &Matrix| model.response(dp);
        let u: Vec<f64> = (0..space.len()).map(|i| (i as f64 * 0.3).cos()).collect();
        let v: Vec<f64> = (0..space.len()).map(|i| (i as f64 * 1.1).sin()).collect();
        let hu = hessian_vector(&u, &space, &mo, &fmo, &resp).unwrap();
        let hv = hessian_vector(&v, &space, &mo, &fmo, &resp).unwrap();
        let uhv = dot(&u, &hv);
        let vhu = dot(&v, &hu);
        assert!(
            (uhv - vhu).abs() < 1.0e-9 * (1.0 + uhv.abs()),
            "Hessian not symmetric: {uhv} vs {vhu}"
        );
    }

    /// Build a model `E[P]=Tr[hP]+¼Tr[PAPA]` whose **exact SCF solution** is a chosen orbital set
    /// `c0` with ascending orbital energies `d` (so the occupied block is the lowest and the orbital
    /// Hessian is positive-definite near the solution). At the solution `F[P(c0)] = c0·diag(d)·c0ᵀ`,
    /// so we set `h = c0·diag(d)·c0ᵀ − ½ A P(c0) A`.
    fn solvable_model(n: usize, nocc: usize) -> (Model, Matrix, Vec<f64>) {
        let a = sym(n, 7);
        let c0 = orthonormal(n, 19);
        let occ: Vec<f64> = (0..n).map(|i| if i < nocc { 2.0 } else { 0.0 }).collect();
        let p0 = density_from_mos(&c0, &occ);
        // Ascending canonical orbital energies (clear occ→virt gap).
        let d: Vec<f64> = (0..n).map(|i| -1.0 + 0.7 * i as f64).collect();
        // fsol = c0 diag(d) c0ᵀ.
        let mut fsol = Matrix::zeros(n, n);
        for mu in 0..n {
            for nu in 0..n {
                let mut s = 0.0;
                for r in 0..n {
                    s += c0[(mu, r)] * d[r] * c0[(nu, r)];
                }
                fsol[(mu, nu)] = s;
            }
        }
        let apa = a.matmul(&p0).unwrap().matmul(&a).unwrap();
        let mut h = Matrix::zeros(n, n);
        for mu in 0..n {
            for nu in 0..n {
                h[(mu, nu)] = fsol[(mu, nu)] - 0.5 * apa[(mu, nu)];
            }
        }
        (Model { h, a }, c0, occ)
    }

    /// **M4.4 gate** — starting from orbitals *perturbed* off a known SCF solution, repeated
    /// trust-region augmented-Hessian Newton steps (matrix-free, CG inner solve + level shift) drive
    /// the orbital-gradient norm down to convergence and lower the energy monotonically. Validates the
    /// whole TRAH macro-iteration: gradient, Hessian–vector, trust-region step, and MO rotation.
    #[test]
    fn trah_iterations_converge_from_perturbed_solution() {
        let (model, c0, occ) = solvable_model(6, 3);
        let space = OrbitalRotationSpace::from_occupations(&occ);
        // Sanity: the gradient at the constructed solution is ~0.
        let fmo_sol = to_mo_basis(&c0, &model.fock(&density_from_mos(&c0, &occ))).unwrap();
        let gsol = norm2(&orbital_gradient(&fmo_sol, &space));
        assert!(
            gsol < 1.0e-9,
            "constructed solution is not stationary: {gsol:.3e}"
        );
        // Perturb the orbitals away from the solution.
        let vpert: Vec<f64> = (0..space.len())
            .map(|i| 0.12 * ((i as f64 * 0.9).sin()))
            .collect();
        let mut mo = rotate_mos(&c0, &space, &vpert).unwrap();
        let energy_at = |m: &Matrix| model.energy(&density_from_mos(m, &occ));
        let grad_at = |m: &Matrix| -> (Vec<f64>, Matrix) {
            let fmo = to_mo_basis(m, &model.fock(&density_from_mos(m, &occ))).unwrap();
            (orbital_gradient(&fmo, &space), fmo)
        };
        let (g_init, _) = grad_at(&mo);
        let g0n = norm2(&g_init);
        assert!(
            g0n > 1.0e-3,
            "perturbation should make the gradient nonzero: {g0n:.3e}"
        );
        let mut e_prev = energy_at(&mo);
        let mut gn = g0n;
        for _ in 0..12 {
            let (g, fmo) = grad_at(&mo);
            gn = norm2(&g);
            if gn < 1.0e-7 {
                break;
            }
            let resp = |dp: &Matrix| model.response(dp);
            let hv = |v: &[f64]| hessian_vector(v, &space, &mo, &fmo, &resp);
            let precond = jacobi_preconditioner(&fmo, &space);
            let step = trust_region_newton_step(&g, &precond, &hv, 0.3, 1.0e-9, 200).unwrap();
            mo = rotate_mos(&mo, &space, &step.kappa).unwrap();
            let e_new = energy_at(&mo);
            assert!(
                e_new <= e_prev + 1.0e-10,
                "energy increased: {e_prev:.8} -> {e_new:.8}"
            );
            e_prev = e_new;
        }
        assert!(
            gn < 1.0e-7,
            "TRAH did not converge the gradient: {g0n:.3e} -> {gn:.3e}"
        );
    }

    /// **M4.5 gate (model)** — the full TRAH macro-iteration driver [`run_trah_scf`] (trust-region
    /// radius adaptation, accept/reject on the actual-vs-predicted ratio) converges from a perturbed
    /// start back to the known SCF solution's energy with a vanishing orbital gradient.
    #[test]
    fn run_trah_scf_converges_to_known_solution() {
        let (model, c0, occ) = solvable_model(6, 3);
        let space = OrbitalRotationSpace::from_occupations(&occ);
        let e_sol = model.energy(&density_from_mos(&c0, &occ));
        let vpert: Vec<f64> = (0..space.len())
            .map(|i| 0.15 * (i as f64 * 0.6).cos())
            .collect();
        let start = rotate_mos(&c0, &space, &vpert).unwrap();
        let fe = |p: &Matrix| (model.fock(p), model.energy(p));
        let fr = |dp: &Matrix| model.response(dp);
        let opt = TrahOptions {
            grad_tol: 1.0e-8,
            ..Default::default()
        };
        let res = run_trah_scf(&start, &occ, fe, fr, &opt).unwrap();
        assert!(
            res.converged,
            "TRAH SCF did not converge (‖g‖={:.3e}, iters={})",
            res.gradient_norm, res.iterations
        );
        assert!(
            res.gradient_norm < 1.0e-8,
            "gradient: {:.3e}",
            res.gradient_norm
        );
        assert!(
            (res.energy - e_sol).abs() < 1.0e-7,
            "energy {:.8} vs known solution {:.8}",
            res.energy,
            e_sol
        );
    }

    /// The O(N³) GEMM `density_response_gemm` and the explicit rank-update produce the same
    /// `δP = Σ_ai v_ai Δn_ai (C_μa C_νi + C_μi C_νa)` to round-off; production [`density_response`]
    /// dispatches between them purely for speed, so they must agree.
    #[test]
    fn density_response_gemm_matches_explicit() {
        let (_, c0, occ) = solvable_model(6, 3);
        let space = OrbitalRotationSpace::from_occupations(&occ);
        let v: Vec<f64> = (0..space.len())
            .map(|i| 0.3 * ((i as f64 + 1.0) * 0.7).sin())
            .collect();
        let a = density_response_explicit(&c0, &space, &v);
        let b = density_response_gemm(&c0, &space, &v);
        let maxdiff = a
            .as_slice()
            .iter()
            .zip(b.as_slice().iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            maxdiff < 1.0e-12,
            "GEMM vs explicit density response differ: {maxdiff:.3e}"
        );
    }
}
