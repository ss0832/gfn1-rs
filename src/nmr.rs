// SPDX-License-Identifier: GPL-3.0-or-later

//! GIAO/LAO closed-shell NMR nuclear magnetic shielding tensors for GFN1-xTB.
//!
//! Target: `sigma_{A,ab} = d^2 E / dB_a dm_{A,b}` for nucleus `A` (`B` = external
//! field, `m_A` = nuclear magnetic moment), a diamagnetic part (expectation of the
//! `B`-`m` second-derivative operator over the unperturbed density) plus a
//! paramagnetic part (the CP-SCC magnetic-field density response contracted with
//! the nuclear magnetic-dipole operator). Non-periodic, closed-shell.
//!
//! Because GFN1-xTB is tight-binding (parameterised `H0`, no explicit
//! electron-nuclear-attraction integrals), this module also hosts the analytic
//! Gaussian Coulomb-type integral machinery the rest of the code does not need:
//! the Boys function and the McMurchie-Davidson nuclear-attraction / `1/r^3`
//! operator integrals over the contracted Cartesian Gaussian basis.
//!
//! Implemented incrementally with finite-difference / quadrature gates at every
//! layer (see the in-module tests).

use crate::basis::BasisSet;
use crate::linalg::Matrix;
use crate::system::PeriodicSystem;

/// Boys function `F_n(x) = ∫_0^1 t^{2n} e^{-x t^2} dt` for `n = 0..=nmax`,
/// returned as `[F_0(x), F_1(x), ..., F_nmax(x)]`. Numerically stable for all
/// `x >= 0`: an ascending series for `F_nmax` followed by the stable downward
/// recurrence `F_{n-1} = (2x F_n + e^{-x})/(2n+1)` for moderate `x`, and the
/// closed form `F_0 = 1/2 sqrt(pi/x)` with upward recurrence for large `x`
/// (`erf(sqrt(x)) = 1` to machine precision there).
///
/// The Boys function is the radial kernel of every nuclear-attraction-type
/// Gaussian integral; the NMR `1/r_A`, `1/r_A^3` and field-gradient operators are
/// built from `F_n` and its McMurchie-Davidson Hermite-Coulomb recurrences.
pub fn boys(nmax: usize, x: f64) -> Vec<f64> {
    debug_assert!(x >= 0.0, "Boys argument must be non-negative");
    let mut f = vec![0.0_f64; nmax + 1];
    if x < 1.0e-13 {
        for (n, fn_val) in f.iter_mut().enumerate() {
            *fn_val = 1.0 / (2.0 * n as f64 + 1.0);
        }
        return f;
    }
    let ex = (-x).exp();
    if x <= 35.0 {
        // Ascending series for F_nmax(x): with n = nmax,
        //   F_n(x) = e^{-x} * sum_{k>=0} (2x)^k * (2n-1)!! / (2n+2k+1)!!
        // evaluated as a running product term_k = (2x)^k / prod_{j=1}^{k}(2n+2j+1)
        // times the k=0 value 1/(2n+1). Converges for all x; fast for x <= 35.
        let n = nmax as f64;
        let mut term = 1.0 / (2.0 * n + 1.0);
        let mut sum = term;
        let mut k = 1.0_f64;
        loop {
            term *= 2.0 * x / (2.0 * n + 2.0 * k + 1.0);
            sum += term;
            if term <= 1.0e-17 * sum {
                break;
            }
            k += 1.0;
            if k > 500.0 {
                break;
            }
        }
        f[nmax] = ex * sum;
        for n in (0..nmax).rev() {
            f[n] = (2.0 * x * f[n + 1] + ex) / (2.0 * n as f64 + 1.0);
        }
    } else {
        f[0] = 0.5 * (std::f64::consts::PI / x).sqrt();
        for n in 1..=nmax {
            f[n] = ((2.0 * n as f64 - 1.0) * f[n - 1] - ex) / (2.0 * x);
        }
    }
    f
}

/// McMurchie-Davidson Hermite expansion coefficients `E^{l1,l2}_t` for one
/// Cartesian direction, returned as `[E_0, ..., E_{l1+l2}]`. Expands the product
/// of two 1D Gaussians `(x-A)^{l1} e^{-α(x-A)^2} (x-B)^{l2} e^{-β(x-B)^2}` in
/// Hermite Gaussians centred at the product centre `P`. `xpa = P-A`, `xpb = P-B`,
/// `p = α+β`, `kab = e^{-μ (A-B)^2}` with `μ = αβ/p`.
pub(crate) fn hermite_e(l1: usize, l2: usize, xpa: f64, xpb: f64, p: f64, kab: f64) -> Vec<f64> {
    let inv2p = 1.0 / (2.0 * p);
    // table[i][j] holds E^{i,j}_t for t = 0..=i+j.
    let mut table: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); l2 + 1]; l1 + 1];
    table[0][0] = vec![kab];
    for i in 1..=l1 {
        let prev = table[i - 1][0].clone();
        let mut cur = vec![0.0_f64; i + 1];
        for (t, slot) in cur.iter_mut().enumerate() {
            let mut val = xpa * prev.get(t).copied().unwrap_or(0.0)
                + (t as f64 + 1.0) * prev.get(t + 1).copied().unwrap_or(0.0);
            if t >= 1 {
                val += inv2p * prev[t - 1];
            }
            *slot = val;
        }
        table[i][0] = cur;
    }
    for i in 0..=l1 {
        for j in 1..=l2 {
            let prev = table[i][j - 1].clone();
            let mut cur = vec![0.0_f64; i + j + 1];
            for (t, slot) in cur.iter_mut().enumerate() {
                let mut val = xpb * prev.get(t).copied().unwrap_or(0.0)
                    + (t as f64 + 1.0) * prev.get(t + 1).copied().unwrap_or(0.0);
                if t >= 1 {
                    val += inv2p * prev.get(t - 1).copied().unwrap_or(0.0);
                }
                *slot = val;
            }
            table[i][j] = cur;
        }
    }
    table[l1][l2].clone()
}

/// Hermite-Coulomb auxiliary integrals `R^0_{tuv}` (McMurchie-Davidson) for the
/// `1/r` operator between a Hermite Gaussian of exponent `p` at `P` and a point
/// charge at `C` (`pc = P - C`). Returned as `r[t][u][v]`, `t<=tmax` etc.
/// Built from `R^N_{000} = (-2p)^N F_N(p |PC|^2)` and the upward recurrences.
pub(crate) fn hermite_coulomb(
    tmax: usize,
    umax: usize,
    vmax: usize,
    p: f64,
    pc: [f64; 3],
) -> Vec<Vec<Vec<f64>>> {
    let nmax = tmax + umax + vmax;
    let r2 = pc[0] * pc[0] + pc[1] * pc[1] + pc[2] * pc[2];
    let fvals = boys(nmax, p * r2);
    let mut r = vec![vec![vec![vec![0.0_f64; vmax + 1]; umax + 1]; tmax + 1]; nmax + 1];
    let mut coef = 1.0;
    for n in 0..=nmax {
        r[n][0][0][0] = coef * fvals[n];
        coef *= -2.0 * p;
    }
    for t in 1..=tmax {
        for n in 0..=(nmax - t) {
            let mut val = pc[0] * r[n + 1][t - 1][0][0];
            if t >= 2 {
                val += (t as f64 - 1.0) * r[n + 1][t - 2][0][0];
            }
            r[n][t][0][0] = val;
        }
    }
    for t in 0..=tmax {
        for u in 1..=umax {
            for n in 0..=(nmax - t - u) {
                let mut val = pc[1] * r[n + 1][t][u - 1][0];
                if u >= 2 {
                    val += (u as f64 - 1.0) * r[n + 1][t][u - 2][0];
                }
                r[n][t][u][0] = val;
            }
        }
    }
    for t in 0..=tmax {
        for u in 0..=umax {
            for v in 1..=vmax {
                for n in 0..=(nmax - t - u - v) {
                    let mut val = pc[2] * r[n + 1][t][u][v - 1];
                    if v >= 2 {
                        val += (v as f64 - 1.0) * r[n + 1][t][u][v - 2];
                    }
                    r[n][t][u][v] = val;
                }
            }
        }
    }
    let mut out = vec![vec![vec![0.0_f64; vmax + 1]; umax + 1]; tmax + 1];
    for (t, ot) in out.iter_mut().enumerate() {
        for (u, ou) in ot.iter_mut().enumerate() {
            for (v, ov) in ou.iter_mut().enumerate() {
                *ov = r[0][t][u][v];
            }
        }
    }
    out
}

/// Nuclear-attraction integral of one primitive Cartesian Gaussian pair with the
/// `1/|r-C|` operator: `∫ (r-A)^la e^{-α|r-A|^2} |r-C|^{-1} (r-B)^lb e^{-β|r-B|^2} d^3r`
/// (raw, un-normalised primitives; contraction coefficients are applied by the
/// caller). `la`/`lb` are Cartesian powers `[lx,ly,lz]`.
fn nuclear_v_primitive(
    la: [usize; 3],
    lb: [usize; 3],
    alpha: f64,
    beta: f64,
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
) -> f64 {
    let p = alpha + beta;
    let mu = alpha * beta / p;
    let pp = [
        (alpha * a[0] + beta * b[0]) / p,
        (alpha * a[1] + beta * b[1]) / p,
        (alpha * a[2] + beta * b[2]) / p,
    ];
    let pc = [pp[0] - c[0], pp[1] - c[1], pp[2] - c[2]];
    let ex = hermite_e(
        la[0],
        lb[0],
        pp[0] - a[0],
        pp[0] - b[0],
        p,
        (-mu * (a[0] - b[0]).powi(2)).exp(),
    );
    let ey = hermite_e(
        la[1],
        lb[1],
        pp[1] - a[1],
        pp[1] - b[1],
        p,
        (-mu * (a[1] - b[1]).powi(2)).exp(),
    );
    let ez = hermite_e(
        la[2],
        lb[2],
        pp[2] - a[2],
        pp[2] - b[2],
        p,
        (-mu * (a[2] - b[2]).powi(2)).exp(),
    );
    let r = hermite_coulomb(la[0] + lb[0], la[1] + lb[1], la[2] + lb[2], p, pc);
    let mut v = 0.0;
    for (t, &et) in ex.iter().enumerate() {
        for (u, &eu) in ey.iter().enumerate() {
            for (w, &ew) in ez.iter().enumerate() {
                v += et * eu * ew * r[t][u][w];
            }
        }
    }
    v * 2.0 * std::f64::consts::PI / p
}

/// Electron-nuclear-attraction matrix `<mu| 1/|r-C| |nu>` over the contracted
/// Cartesian Gaussian AO basis, for a point at `c` (atomic units / Bohr). This is
/// the building block for the GIAO NMR `1/r^3` magnetic-dipole operators. The
/// physical `V_ne` contribution would be `-Z_C` times this; here we return the
/// bare `1/r_C` integral.
pub fn nuclear_attraction_matrix(system: &PeriodicSystem, basis: &BasisSet, c: [f64; 3]) -> Matrix {
    let n = basis.len();
    let mut mat = Matrix::zeros(n, n);
    for i in 0..n {
        let ao_i = &basis.aos[i];
        let ra = system.atoms[ao_i.atom_index].position;
        let a = [ra.x, ra.y, ra.z];
        for j in 0..=i {
            let ao_j = &basis.aos[j];
            let rb = system.atoms[ao_j.atom_index].position;
            let b = [rb.x, rb.y, rb.z];
            let mut v = 0.0;
            for ci in &ao_i.components {
                let la = [ci.power.x, ci.power.y, ci.power.z];
                for pi in &ao_i.primitives {
                    let wa = ci.coefficient * pi.coefficient;
                    for cj in &ao_j.components {
                        let lb = [cj.power.x, cj.power.y, cj.power.z];
                        for pj in &ao_j.primitives {
                            v += wa
                                * cj.coefficient
                                * pj.coefficient
                                * nuclear_v_primitive(la, lb, pi.exponent, pj.exponent, a, b, c);
                        }
                    }
                }
            }
            mat[(i, j)] = v;
            mat[(j, i)] = v;
        }
    }
    mat
}

/// Electric-field-type integral `<la| (r-C)_j / |r-C|^3 |lb>` for one primitive
/// Cartesian Gaussian pair and component `j ∈ {0,1,2}`. Equals `∂/∂C_j` of the
/// nuclear-attraction integral, i.e. `-2π/p Σ E_t E_u E_v R^0_{(tuv)+e_j}` (one
/// extra Hermite order in direction `j`). This `1/r^3` integral is the building
/// block of the NMR paramagnetic nuclear magnetic-dipole operator.
fn field_v_primitive(
    la: [usize; 3],
    lb: [usize; 3],
    j: usize,
    alpha: f64,
    beta: f64,
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
) -> f64 {
    let p = alpha + beta;
    let mu = alpha * beta / p;
    let pp = [
        (alpha * a[0] + beta * b[0]) / p,
        (alpha * a[1] + beta * b[1]) / p,
        (alpha * a[2] + beta * b[2]) / p,
    ];
    let pc = [pp[0] - c[0], pp[1] - c[1], pp[2] - c[2]];
    let ex = hermite_e(
        la[0],
        lb[0],
        pp[0] - a[0],
        pp[0] - b[0],
        p,
        (-mu * (a[0] - b[0]).powi(2)).exp(),
    );
    let ey = hermite_e(
        la[1],
        lb[1],
        pp[1] - a[1],
        pp[1] - b[1],
        p,
        (-mu * (a[1] - b[1]).powi(2)).exp(),
    );
    let ez = hermite_e(
        la[2],
        lb[2],
        pp[2] - a[2],
        pp[2] - b[2],
        p,
        (-mu * (a[2] - b[2]).powi(2)).exp(),
    );
    let (dt, du, dv) = match j {
        0 => (1usize, 0usize, 0usize),
        1 => (0, 1, 0),
        _ => (0, 0, 1),
    };
    let r = hermite_coulomb(
        la[0] + lb[0] + dt,
        la[1] + lb[1] + du,
        la[2] + lb[2] + dv,
        p,
        pc,
    );
    let mut v = 0.0;
    for (t, &et) in ex.iter().enumerate() {
        for (u, &eu) in ey.iter().enumerate() {
            for (w, &ew) in ez.iter().enumerate() {
                v += et * eu * ew * r[t + dt][u + du][w + dv];
            }
        }
    }
    -v * 2.0 * std::f64::consts::PI / p
}

/// The three Cartesian components of the electric-field integral
/// `<mu| (r-C)/|r-C|^3 |nu>` over the contracted AO basis (the electron-density
/// electric field at point `c`). Returned as `[V_x, V_y, V_z]`. Foundation of the
/// NMR paramagnetic operator `(r_C x grad)/r_C^3`.
pub fn electric_field_integrals(
    system: &PeriodicSystem,
    basis: &BasisSet,
    c: [f64; 3],
) -> [Matrix; 3] {
    let n = basis.len();
    let mut out = [
        Matrix::zeros(n, n),
        Matrix::zeros(n, n),
        Matrix::zeros(n, n),
    ];
    for i in 0..n {
        let ao_i = &basis.aos[i];
        let ra = system.atoms[ao_i.atom_index].position;
        let a = [ra.x, ra.y, ra.z];
        for j in 0..n {
            let ao_j = &basis.aos[j];
            let rb = system.atoms[ao_j.atom_index].position;
            let b = [rb.x, rb.y, rb.z];
            let mut acc = [0.0_f64; 3];
            for ci in &ao_i.components {
                let la = [ci.power.x, ci.power.y, ci.power.z];
                for pi in &ao_i.primitives {
                    let wa = ci.coefficient * pi.coefficient;
                    for cj in &ao_j.components {
                        let lb = [cj.power.x, cj.power.y, cj.power.z];
                        for pj in &ao_j.primitives {
                            let w = wa * cj.coefficient * pj.coefficient;
                            for (k, slot) in acc.iter_mut().enumerate() {
                                *slot += w * field_v_primitive(
                                    la,
                                    lb,
                                    k,
                                    pi.exponent,
                                    pj.exponent,
                                    a,
                                    b,
                                    c,
                                );
                            }
                        }
                    }
                }
            }
            for (k, m) in out.iter_mut().enumerate() {
                m[(i, j)] = acc[k];
            }
        }
    }
    out
}

/// The two `(j_dir, k_dir, sign)` triples of the cross product `(r x grad)_bcomp`.
fn curl_terms(bcomp: usize) -> [(usize, usize, f64); 2] {
    match bcomp {
        0 => [(1, 2, 1.0), (2, 1, -1.0)], // L_x = y d_z - z d_y
        1 => [(2, 0, 1.0), (0, 2, -1.0)], // L_y = z d_x - x d_z
        _ => [(0, 1, 1.0), (1, 0, -1.0)], // L_z = x d_y - y d_x
    }
}

/// Paramagnetic nuclear magnetic-dipole operator integral
/// `<la| (r_C x grad)_bcomp / |r-C|^3 |lb>` for one primitive pair (real,
/// antisymmetric). `(r_C x grad)_b = sum_{jk} eps_{bjk} r_{C,j} d_k`, and each
/// `<la| r_{C,j}/r_C^3 d_k |lb>` uses the ket-derivative relation
/// `d_k|lb> = lb_k|lb-e_k> - 2β|lb+e_k>` on the 2a field integral.
fn para_v_primitive(
    la: [usize; 3],
    lb: [usize; 3],
    bcomp: usize,
    alpha: f64,
    beta: f64,
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
) -> f64 {
    let mut total = 0.0;
    for (jdir, kdir, sign) in curl_terms(bcomp) {
        let mut s = 0.0;
        if lb[kdir] >= 1 {
            let mut lbm = lb;
            lbm[kdir] -= 1;
            s += lb[kdir] as f64 * field_v_primitive(la, lbm, jdir, alpha, beta, a, b, c);
        }
        let mut lbp = lb;
        lbp[kdir] += 1;
        s -= 2.0 * beta * field_v_primitive(la, lbp, jdir, alpha, beta, a, b, c);
        total += sign * s;
    }
    total
}

/// The three Cartesian components of the paramagnetic nuclear operator matrix
/// `<mu| (r_C x grad)_b / |r-C|^3 |nu>` over the contracted AO basis, for nucleus
/// position `c`. Real and antisymmetric. The physical paramagnetic operator is
/// `-i` times this; combined with the (imaginary) CP-SCC magnetic-field density
/// response it yields the real paramagnetic shielding (assembled in a later step).
pub fn paramagnetic_operator_matrix(
    system: &PeriodicSystem,
    basis: &BasisSet,
    c: [f64; 3],
) -> [Matrix; 3] {
    let n = basis.len();
    let mut out = [
        Matrix::zeros(n, n),
        Matrix::zeros(n, n),
        Matrix::zeros(n, n),
    ];
    for i in 0..n {
        let ao_i = &basis.aos[i];
        let ra = system.atoms[ao_i.atom_index].position;
        let a = [ra.x, ra.y, ra.z];
        for j in 0..n {
            let ao_j = &basis.aos[j];
            let rb = system.atoms[ao_j.atom_index].position;
            let b = [rb.x, rb.y, rb.z];
            let mut acc = [0.0_f64; 3];
            for ci in &ao_i.components {
                let la = [ci.power.x, ci.power.y, ci.power.z];
                for pi in &ao_i.primitives {
                    let wa = ci.coefficient * pi.coefficient;
                    for cj in &ao_j.components {
                        let lb = [cj.power.x, cj.power.y, cj.power.z];
                        for pj in &ao_j.primitives {
                            let w = wa * cj.coefficient * pj.coefficient;
                            for (bcomp, slot) in acc.iter_mut().enumerate() {
                                *slot += w * para_v_primitive(
                                    la,
                                    lb,
                                    bcomp,
                                    pi.exponent,
                                    pj.exponent,
                                    a,
                                    b,
                                    c,
                                );
                            }
                        }
                    }
                }
            }
            for (bcomp, m) in out.iter_mut().enumerate() {
                m[(i, j)] = acc[bcomp];
            }
        }
    }
    out
}

/// Second-moment `1/r^3` integral `<la| (r-C)_adir (r-C)_bdir / |r-C|^3 |lb>`,
/// built from the 2a field integral by raising the bra power in `adir`:
/// `(r-C)_adir = (r-A)_adir + (A-C)_adir`. Satisfies the trace identity
/// `sum_adir M_{adir,adir} = <la|1/r_C|lb>` (since `sum_k (r-C)_k^2/r_C^3 = 1/r_C`).
fn second_moment_field_primitive(
    la: [usize; 3],
    lb: [usize; 3],
    adir: usize,
    bdir: usize,
    alpha: f64,
    beta: f64,
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
) -> f64 {
    let mut lap = la;
    lap[adir] += 1;
    field_v_primitive(lap, lb, bdir, alpha, beta, a, b, c)
        + (a[adir] - c[adir]) * field_v_primitive(la, lb, bdir, alpha, beta, a, b, c)
}

/// Diamagnetic NMR shielding operator for one primitive pair: the 3x3 array
/// `d[a][b] = <la| [(r_O.r_C) δ_ab - r_{O,a} r_{C,b}] / |r-C|^3 |lb>` for nucleus
/// `c` and gauge origin `o`. All pieces reduce to the verified nuclear/field/
/// second-moment integrals: `(r_O.r_C)/r_C^3 = 1/r_C + Σ_k(C_k-O_k) field_k`, and
/// `r_{O,a} r_{C,b}/r_C^3 = secondmoment_{ab} + (C_a-O_a) field_b`.
fn dia_v_primitive(
    la: [usize; 3],
    lb: [usize; 3],
    alpha: f64,
    beta: f64,
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
    o: [f64; 3],
) -> [[f64; 3]; 3] {
    let nuc = nuclear_v_primitive(la, lb, alpha, beta, a, b, c);
    let mut field = [0.0_f64; 3];
    for (k, fk) in field.iter_mut().enumerate() {
        *fk = field_v_primitive(la, lb, k, alpha, beta, a, b, c);
    }
    let rodotrc = nuc + (0..3).map(|k| (c[k] - o[k]) * field[k]).sum::<f64>();
    let mut d = [[0.0_f64; 3]; 3];
    for (adir, drow) in d.iter_mut().enumerate() {
        for (bdir, dval) in drow.iter_mut().enumerate() {
            let sm = second_moment_field_primitive(la, lb, adir, bdir, alpha, beta, a, b, c);
            let r_oa_rcb = sm + (c[adir] - o[adir]) * field[bdir];
            *dval = if adir == bdir { rodotrc } else { 0.0 } - r_oa_rcb;
        }
    }
    d
}

/// Diamagnetic NMR shielding operator matrices
/// `d_ab[mu][nu] = <mu| [(r_O.r_C) δ_ab - r_{O,a} r_{C,b}] / |r-C|^3 |nu>` over the
/// contracted AO basis, for nucleus position `c` and gauge origin `o`. Returned as
/// `d[a][b]` (3x3 array of AO matrices). With the unperturbed density it gives the
/// diamagnetic shielding `σ^dia_{A,ab} = Σ_{mu nu} P_{mu nu} d_ab[mu][nu]` (times the
/// physical prefactor; assembled with the GIAO gauge in a later step).
pub fn diamagnetic_operator_matrix(
    system: &PeriodicSystem,
    basis: &BasisSet,
    c: [f64; 3],
    o: [f64; 3],
) -> [[Matrix; 3]; 3] {
    let n = basis.len();
    let mut out: [[Matrix; 3]; 3] =
        std::array::from_fn(|_| std::array::from_fn(|_| Matrix::zeros(n, n)));
    for i in 0..n {
        let ao_i = &basis.aos[i];
        let ra = system.atoms[ao_i.atom_index].position;
        let a = [ra.x, ra.y, ra.z];
        for j in 0..n {
            let ao_j = &basis.aos[j];
            let rb = system.atoms[ao_j.atom_index].position;
            let b = [rb.x, rb.y, rb.z];
            let mut acc = [[0.0_f64; 3]; 3];
            for ci in &ao_i.components {
                let la = [ci.power.x, ci.power.y, ci.power.z];
                for pi in &ao_i.primitives {
                    let wa = ci.coefficient * pi.coefficient;
                    for cj in &ao_j.components {
                        let lb = [cj.power.x, cj.power.y, cj.power.z];
                        for pj in &ao_j.primitives {
                            let w = wa * cj.coefficient * pj.coefficient;
                            let d = dia_v_primitive(la, lb, pi.exponent, pj.exponent, a, b, c, o);
                            for adir in 0..3 {
                                for bdir in 0..3 {
                                    acc[adir][bdir] += w * d[adir][bdir];
                                }
                            }
                        }
                    }
                }
            }
            for adir in 0..3 {
                for bdir in 0..3 {
                    out[adir][bdir][(i, j)] = acc[adir][bdir];
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// High-accuracy reference: composite Simpson of `∫_0^1 t^{2n} e^{-x t^2} dt`.
    fn boys_quadrature(n: usize, x: f64) -> f64 {
        let m = 8000usize; // even
        let h = 1.0 / m as f64;
        let integrand = |t: f64| t.powi(2 * n as i32) * (-x * t * t).exp();
        let mut s = integrand(0.0) + integrand(1.0);
        for i in 1..m {
            let t = i as f64 * h;
            s += if i % 2 == 0 { 2.0 } else { 4.0 } * integrand(t);
        }
        s * h / 3.0
    }

    #[test]
    fn boys_at_zero_is_exact() {
        let f = boys(8, 0.0);
        for (n, &v) in f.iter().enumerate() {
            assert!(
                (v - 1.0 / (2.0 * n as f64 + 1.0)).abs() < 1.0e-15,
                "F_{n}(0) = {v}"
            );
        }
    }

    #[test]
    fn boys_matches_quadrature() {
        // covers the small-x series branch and the large-x (x>35) closed-form branch
        for &x in &[1.0e-6, 0.25, 1.0, 5.0, 15.0, 30.0, 40.0, 80.0] {
            let f = boys(8, x);
            for n in 0..=8 {
                let q = boys_quadrature(n, x);
                let tol = 1.0e-10 + 1.0e-8 * q.abs();
                assert!(
                    (f[n] - q).abs() < tol,
                    "boys({n},{x}) = {} vs quadrature {q} (tol {tol:.1e})",
                    f[n]
                );
            }
        }
    }

    #[test]
    fn boys_f0_closed_form() {
        // F_0(x) = (1/2) sqrt(pi/x) erf(sqrt(x)); spot-check a moderate x where
        // erf is not yet saturated, using the series-relation erf via quadrature.
        let x = 2.0_f64;
        let f0 = boys(0, x)[0];
        let q = boys_quadrature(0, x);
        assert!((f0 - q).abs() < 1.0e-12, "F_0({x}) = {f0} vs {q}");
    }

    /// The McMurchie-Davidson s-s nuclear-attraction integral must equal the closed
    /// form `(2π/p) e^{-μ|A-B|^2} F_0(p|P-C|^2)` — verifies the E^{00}/R_{000}/prefactor
    /// path independently of the recurrences.
    #[test]
    fn nuclear_ss_closed_form() {
        let a = [0.10, 0.20, -0.30];
        let b = [0.40, -0.10, 0.50];
        let c = [-0.20, 0.30, 0.10];
        let (alpha, beta) = (1.3_f64, 0.7_f64);
        let v = nuclear_v_primitive([0, 0, 0], [0, 0, 0], alpha, beta, a, b, c);
        let p = alpha + beta;
        let mu = alpha * beta / p;
        let pp = [
            (alpha * a[0] + beta * b[0]) / p,
            (alpha * a[1] + beta * b[1]) / p,
            (alpha * a[2] + beta * b[2]) / p,
        ];
        let ab2 = (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>();
        let pc2 = (0..3).map(|k| (pp[k] - c[k]).powi(2)).sum::<f64>();
        let closed = 2.0 * std::f64::consts::PI / p * (-mu * ab2).exp() * boys(0, p * pc2)[0];
        assert!((v - closed).abs() < 1.0e-13, "{v} vs {closed}");
    }

    /// Higher angular momentum is verified rigorously and quadrature-free via the
    /// exact center-derivative relation `∂/∂A_x V(l_x) = -l_x V(l_x-1) + 2α V(l_x+1)`
    /// (and the analogue on B/β), which exercises the full E and R_{tuv} recurrences.
    #[test]
    fn nuclear_angular_momentum_via_center_fd() {
        let a = [0.10, 0.20, -0.30];
        let b = [0.40, -0.10, 0.50];
        let c = [-0.20, 0.30, 0.10];
        let (alpha, beta) = (1.3_f64, 0.7_f64);
        let h = 1.0e-5;
        let vp = |la: [usize; 3], lb: [usize; 3], ai: [f64; 3], bi: [f64; 3]| {
            nuclear_v_primitive(la, lb, alpha, beta, ai, bi, c)
        };
        // p_x bra: V([1,0,0]) = (1/2α) ∂/∂A_x V([0,0,0])
        let mut ap = a;
        ap[0] += h;
        let mut am = a;
        am[0] -= h;
        let dvdax = (vp([0, 0, 0], [0, 0, 0], ap, b) - vp([0, 0, 0], [0, 0, 0], am, b)) / (2.0 * h);
        let vpx = vp([1, 0, 0], [0, 0, 0], a, b);
        assert!(
            (vpx - dvdax / (2.0 * alpha)).abs() < 1.0e-7,
            "px {vpx} vs {}",
            dvdax / (2.0 * alpha)
        );
        // p_y ket: V(_,[0,1,0]) = (1/2β) ∂/∂B_y V(_,[0,0,0])
        let mut bp = b;
        bp[1] += h;
        let mut bm = b;
        bm[1] -= h;
        let dvdby = (vp([0, 0, 0], [0, 0, 0], a, bp) - vp([0, 0, 0], [0, 0, 0], a, bm)) / (2.0 * h);
        let vpy = vp([0, 0, 0], [0, 1, 0], a, b);
        assert!((vpy - dvdby / (2.0 * beta)).abs() < 1.0e-7);
        // d_xx bra: V([2,0,0]) = (∂/∂A_x V([1,0,0]) + V([0,0,0])) / (2α)
        let dv1dax =
            (vp([1, 0, 0], [0, 0, 0], ap, b) - vp([1, 0, 0], [0, 0, 0], am, b)) / (2.0 * h);
        let v0 = vp([0, 0, 0], [0, 0, 0], a, b);
        let vdxx = vp([2, 0, 0], [0, 0, 0], a, b);
        assert!(
            (vdxx - (dv1dax + v0) / (2.0 * alpha)).abs() < 1.0e-6,
            "dxx {vdxx}"
        );
        // mixed d_xy bra and a p-p cross term, via the same relation chain
        let dvy_for_x =
            (vp([0, 1, 0], [0, 0, 0], ap, b) - vp([0, 1, 0], [0, 0, 0], am, b)) / (2.0 * h);
        let vdxy = vp([1, 1, 0], [0, 0, 0], a, b);
        assert!(
            (vdxy - dvy_for_x / (2.0 * alpha)).abs() < 1.0e-6,
            "dxy {vdxy}"
        );
    }

    /// The 1/r^3 electric-field integral must equal the C-derivative of the
    /// nuclear-attraction integral: `<la| (r-C)_j/r_C^3 |lb> == ∂/∂C_j <la|1/r_C|lb>`.
    /// Checked for s/p/d bra-ket combinations and all three field components.
    #[test]
    fn field_integral_is_c_derivative_of_nuclear() {
        let a = [0.10, 0.20, -0.30];
        let b = [0.40, -0.10, 0.50];
        let c = [-0.20, 0.30, 0.10];
        let (alpha, beta) = (1.3_f64, 0.7_f64);
        let h = 1.0e-5;
        let cases: [([usize; 3], [usize; 3]); 5] = [
            ([0, 0, 0], [0, 0, 0]),
            ([1, 0, 0], [0, 0, 0]),
            ([0, 0, 0], [0, 1, 0]),
            ([1, 0, 0], [0, 0, 1]),
            ([2, 0, 0], [0, 1, 0]),
        ];
        for (la, lb) in cases {
            for j in 0..3 {
                let mut cp = c;
                cp[j] += h;
                let mut cm = c;
                cm[j] -= h;
                let fd = (nuclear_v_primitive(la, lb, alpha, beta, a, b, cp)
                    - nuclear_v_primitive(la, lb, alpha, beta, a, b, cm))
                    / (2.0 * h);
                let analytic = field_v_primitive(la, lb, j, alpha, beta, a, b, c);
                assert!(
                    (analytic - fd).abs() < 1.0e-6,
                    "field({la:?},{lb:?},j={j}) = {analytic} vs FD {fd}"
                );
            }
        }
    }

    /// The paramagnetic operator `<la|(r_C x grad)_b/r_C^3|lb>` must equal
    /// `-sum_{jk} eps_{bjk} ∂/∂B_k field_j(la,lb)` (the ket gradient equals minus the
    /// ket-centre derivative), verifying the raise/lower ket-derivative algebra.
    #[test]
    fn paramagnetic_operator_via_field_fd() {
        let a = [0.10, 0.20, -0.30];
        let b = [0.40, -0.10, 0.50];
        let c = [-0.20, 0.30, 0.10];
        let (alpha, beta) = (1.3_f64, 0.7_f64);
        let h = 1.0e-5;
        let cases: [([usize; 3], [usize; 3]); 4] = [
            ([0, 0, 0], [0, 0, 0]),
            ([1, 0, 0], [0, 1, 0]),
            ([0, 0, 1], [1, 0, 0]),
            ([1, 1, 0], [0, 0, 1]),
        ];
        for (la, lb) in cases {
            for bcomp in 0..3 {
                let mut fd_total = 0.0;
                for (jdir, kdir, sign) in curl_terms(bcomp) {
                    let mut bp = b;
                    bp[kdir] += h;
                    let mut bm = b;
                    bm[kdir] -= h;
                    let dfield = (field_v_primitive(la, lb, jdir, alpha, beta, a, bp, c)
                        - field_v_primitive(la, lb, jdir, alpha, beta, a, bm, c))
                        / (2.0 * h);
                    fd_total += sign * (-dfield);
                }
                let analytic = para_v_primitive(la, lb, bcomp, alpha, beta, a, b, c);
                assert!(
                    (analytic - fd_total).abs() < 1.0e-6,
                    "para({la:?},{lb:?},b={bcomp}) = {analytic} vs FD {fd_total}"
                );
            }
        }
    }

    /// The orbital angular-momentum operator about C is anti-Hermitian, so its real
    /// primitive matrix element is antisymmetric under bra<->ket exchange:
    /// `<a|(r_C x grad)_b|a'> = -<a'|(r_C x grad)_b|a>`.
    #[test]
    fn paramagnetic_operator_is_antisymmetric() {
        let a = [0.10, 0.20, -0.30];
        let b = [0.40, -0.10, 0.50];
        let c = [-0.20, 0.30, 0.10];
        let (alpha, beta) = (1.3_f64, 0.7_f64);
        let cases: [([usize; 3], [usize; 3]); 4] = [
            ([0, 0, 0], [1, 0, 0]),
            ([1, 0, 0], [0, 1, 0]),
            ([0, 1, 0], [0, 0, 1]),
            ([1, 1, 0], [0, 0, 1]),
        ];
        for (la, lb) in cases {
            for bcomp in 0..3 {
                let forward = para_v_primitive(la, lb, bcomp, alpha, beta, a, b, c);
                let reversed = para_v_primitive(lb, la, bcomp, beta, alpha, b, a, c);
                assert!(
                    (forward + reversed).abs() < 1.0e-11,
                    "antisymmetry b={bcomp}: {forward} vs {reversed}"
                );
            }
        }
    }

    /// The second-moment 1/r^3 integral must satisfy the exact trace identity
    /// `sum_a <la|(r-C)_a^2/r_C^3|lb> = <la|1/r_C|lb>` (verified nuclear integral) and
    /// be symmetric in (a,b). This ties the diamagnetic building block to 1b.
    #[test]
    fn second_moment_trace_and_symmetry() {
        let a = [0.10, 0.20, -0.30];
        let b = [0.40, -0.10, 0.50];
        let c = [-0.20, 0.30, 0.10];
        let (alpha, beta) = (1.3_f64, 0.7_f64);
        let cases: [([usize; 3], [usize; 3]); 4] = [
            ([0, 0, 0], [0, 0, 0]),
            ([1, 0, 0], [0, 1, 0]),
            ([0, 0, 1], [1, 0, 0]),
            ([1, 1, 0], [0, 0, 1]),
        ];
        for (la, lb) in cases {
            let trace: f64 = (0..3)
                .map(|k| second_moment_field_primitive(la, lb, k, k, alpha, beta, a, b, c))
                .sum();
            let nuc = nuclear_v_primitive(la, lb, alpha, beta, a, b, c);
            assert!(
                (trace - nuc).abs() < 1.0e-10,
                "trace {trace} vs nuclear {nuc}"
            );
            for adir in 0..3 {
                for bdir in 0..3 {
                    let m_ab =
                        second_moment_field_primitive(la, lb, adir, bdir, alpha, beta, a, b, c);
                    let m_ba =
                        second_moment_field_primitive(la, lb, bdir, adir, alpha, beta, a, b, c);
                    assert!((m_ab - m_ba).abs() < 1.0e-12, "sym ({adir},{bdir})");
                }
            }
        }
    }

    /// The diamagnetic operator trace `sum_a d_aa = 2 (r_O.r_C)/r_C^3` (since
    /// `sum_a [δ_aa(r_O.r_C) - r_{O,a}r_{C,a}]/r_C^3 = (3-1)(r_O.r_C)/r_C^3`), and the
    /// `(r_O.r_C)/r_C^3` part = `<la|1/r_C|lb> + Σ_k(C_k-O_k) field_k`.
    #[test]
    fn diamagnetic_operator_trace() {
        let a = [0.10, 0.20, -0.30];
        let b = [0.40, -0.10, 0.50];
        let c = [-0.20, 0.30, 0.10];
        let o = [0.05, -0.15, 0.25];
        let (alpha, beta) = (1.3_f64, 0.7_f64);
        for (la, lb) in [([0usize, 0, 0], [0usize, 0, 0]), ([1, 0, 0], [0, 1, 0])] {
            let d = dia_v_primitive(la, lb, alpha, beta, a, b, c, o);
            let trace = d[0][0] + d[1][1] + d[2][2];
            let nuc = nuclear_v_primitive(la, lb, alpha, beta, a, b, c);
            let rodotrc = nuc
                + (0..3)
                    .map(|k| (c[k] - o[k]) * field_v_primitive(la, lb, k, alpha, beta, a, b, c))
                    .sum::<f64>();
            assert!(
                (trace - 2.0 * rodotrc).abs() < 1.0e-10,
                "dia trace {trace} vs {}",
                2.0 * rodotrc
            );
        }
    }
}
