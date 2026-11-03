// SPDX-License-Identifier: GPL-3.0-or-later
//! Exact two-body D3 chain in (R,CN) space. C6 pair interpolation needs only
//! two independent CN variables, not a full 3N-dimensional jet per atom pair.
//! Keep O(N*(3N)^2) CN jets and compact C6 partials instead of O(N^2*(3N)^2).
use super::*;

fn weights_in_dimension(weights: &[Jet2], axis: usize) -> Vec<Jet2> {
    weights
        .iter()
        .map(|w| {
            let mut result = Jet2::constant(w.value, 2);
            result.gradient[axis] = w.gradient[0];
            result.hessian[2 * axis + axis] = w.hessian[0];
            result
        })
        .collect()
}
pub(super) fn two_body(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    path: Option<&str>,
) -> Result<DispersionHessianResult> {
    let s6 = params.global("s6", 1.);
    let s8 = params.required_global("s8")?;
    let a1 = params.required_global("a1")?;
    let a2 = params.required_global("a2")?;
    let reference = resolve_and_load_d3_reference(path)?;
    let n = system.atoms.len();
    let d = 3 * n;
    let cn = d3_coordination_jets::<Jet2>(system, d)?;
    let single: Vec<_> = cn
        .iter()
        .map(|c| {
            let mut x = Jet2::constant(c.value, 1);
            x.gradient[0] = 1.;
            x
        })
        .collect();
    let weights = reference_weight_jets(system, &reference, &single, 1)?;
    let first: Vec<_> = weights.iter().map(|w| weights_in_dimension(w, 0)).collect();
    let second: Vec<_> = weights.iter().map(|w| weights_in_dimension(w, 1)).collect();
    // value, d_i, d_j, d_ii, d_ij, d_jj. Transposed entries swap CN axes.
    let mut partial = vec![vec![[0.; 6]; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let c = atomic_c6_pair_jet(
                &reference,
                &first[i],
                &second[j],
                system.atoms[i].z,
                system.atoms[j].z,
                2,
            )?;
            let p = [
                c.value,
                c.gradient[0],
                c.gradient[1],
                c.hessian[0],
                c.hessian[1],
                c.hessian[3],
            ];
            partial[i][j] = p;
            partial[j][i] = [p[0], p[2], p[1], p[5], p[4], p[3]];
        }
    }
    let mut energy = 0.;
    let mut gradient = vec![Vec3::zero(); n];
    let mut hessian = Matrix::zeros(d, d);
    let mut de = vec![0.; n];
    let mut d2e = Matrix::zeros(n, n);
    let mut mixed = Matrix::zeros(d, n);
    for pair in unique_short_range_pairs(system, DISP2_CUTOFF)? {
        let i = pair.i;
        let j = pair.j;
        let value = pair.dr.norm2();
        if value <= DIST_EPS || value > DISP2_CUTOFF * DISP2_CUTOFF {
            continue;
        }
        let r = value.sqrt();
        let mut r2 = Jet2::constant(value, 1);
        r2.gradient[0] = 2. * r;
        r2.hessian[0] = 2.;
        let scale = 3. * reference.r4r2(system.atoms[i].z)? * reference.r4r2(system.atoms[j].z)?;
        let r0 = a1 * scale.sqrt() + a2;
        let r0_2 = r0 * r0;
        let r0_6 = r0_2 * r0_2 * r0_2;
        let r0_8 = r0_6 * r0_2;
        let r4 = r2.mul(&r2);
        let damping = r4
            .mul(&r2)
            .add_scalar(r0_6)
            .powf(-1.)
            .scale(s6)
            .add(&r4.mul(&r4).add_scalar(r0_8).powf(-1.).scale(s8 * scale));
        let p = partial[i][j];
        let b = damping.value;
        let b1 = damping.gradient[0];
        let b2 = damping.hessian[0];
        energy -= p[0] * b;
        de[i] -= p[1] * b;
        de[j] -= p[2] * b;
        d2e[(i, i)] -= p[3] * b;
        d2e[(i, j)] -= p[4] * b;
        d2e[(j, i)] -= p[4] * b;
        d2e[(j, j)] -= p[5] * b;
        if i == j {
            continue;
        }
        let unit = pair.dr / r;
        let vector = unit * (-p[0] * b1);
        gradient[i] -= vector;
        gradient[j] += vector;
        let unit = unit.to_array();
        for a in 0..3 {
            let ci = -p[1] * b1 * unit[a];
            let cj = -p[2] * b1 * unit[a];
            mixed[(3 * i + a, i)] -= ci;
            mixed[(3 * j + a, i)] += ci;
            mixed[(3 * i + a, j)] -= cj;
            mixed[(3 * j + a, j)] += cj;
            for axis in 0..3 {
                let block = -p[0]
                    * ((b2 - b1 / r) * unit[a] * unit[axis] + if a == axis { b1 / r } else { 0. });
                for (x, y, sign) in [(i, i, 1.), (j, j, 1.), (i, j, -1.), (j, i, -1.)] {
                    hessian[(3 * x + a, 3 * y + axis)] += sign * block;
                }
            }
        }
    }
    let jacobian = Matrix::from_vec(
        n,
        d,
        cn.iter().flat_map(|c| c.gradient.iter().copied()).collect(),
    )?;
    let indirect = jacobian.transpose().matmul(&d2e.matmul(&jacobian)?)?;
    let cross = mixed.matmul(&jacobian)?;
    for x in 0..d {
        let chain: f64 = (0..n).map(|i| de[i] * cn[i].gradient[x]).sum();
        match x % 3 {
            0 => gradient[x / 3].x += chain,
            1 => gradient[x / 3].y += chain,
            _ => gradient[x / 3].z += chain,
        }
        for y in 0..d {
            let geometric: f64 = (0..n).map(|i| de[i] * cn[i].hessian[x * d + y]).sum();
            hessian[(x, y)] += indirect[(x, y)] + cross[(x, y)] + cross[(y, x)] + geometric;
        }
    }
    let stress = if system.lattice.is_some() {
        Some(periodic_dispersion_stress(system, params, path)?)
    } else {
        None
    };
    Ok(DispersionHessianResult {
        energy,
        gradient,
        hessian,
        stress,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reduced_cn_chain_matches_independent_full_geometry_jets() {
        let params = Gfn1Parameters::builtin().unwrap();
        for xyz in [
            "6\nmixed molecule\nC 0 0 0\nH .9 .2 .1\nN -.5 1.3 .2\nO 1.8 -.4 .3\nCl -.7 -.8 1.4\nBr 2.4 1.1 -.5\n",
            "3\nLattice=\"5 0 0 .2 5.1 0 0 .1 5.2\" pbc=\"T T T\"\nO .1 .2 0\nH .95 .73 .03\nH -.64 .86 -.04\n",
            "1\nLattice=\"3 0 0 0 3 0 0 0 3\" pbc=\"T T T\"\nC 0 0 0\n",
        ]{
            let system=PeriodicSystem::from_xyz_str(xyz,0.,false).unwrap();
            let reduced=two_body(&system,&params,None).unwrap();let global=dispersion_hessian_global_jets(&system,&params,None).unwrap();
            assert!((reduced.energy-global.energy).abs()<2e-13);
            for(a,b)in reduced.gradient.iter().zip(global.gradient){assert!((*a-b).norm()<2e-12);}
            let delta=reduced.hessian.as_slice().iter().zip(global.hessian.as_slice()).map(|(a,b)|(a-b).abs()).fold(0.,f64::max);
            assert!(delta<2e-11,"reduced/global D3 Hessian difference {delta:e}");
        }
    }
}
