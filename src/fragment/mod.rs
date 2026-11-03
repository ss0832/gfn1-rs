// SPDX-License-Identifier: GPL-3.0-or-later
//! Embedded fragment molecular orbitals and overlapping divide-and-conquer.
//! All reported derivatives differentiate the stated finite-fragment energy,
//! including the self-consistent embedding response.
mod cluster;
mod dc;
mod geometry;
pub(crate) mod spectral;

use crate::electronic::{BroydenMixer, ElectronicOptions};
use crate::error::{Gfn1Error, Result};
use crate::linalg::{matrix_vector_product, DenseLu, Matrix};
use crate::math::Vec3;
use crate::params::Gfn1Parameters;
use crate::pbc::PbcOptions;
use crate::system::PeriodicSystem;
use cluster::{Cluster, LocalState, Screened};
use geometry::Geometry;

pub use dc::{divide_and_conquer, DivideConquerOptions};

/// Oriented covalent boundary: detach one electron at `bda`, retain its bonding
/// hybrid as a ghost in the fragment containing `baa`. Indices are zero-based.
/// The integer image is fixed when differentiating a periodic cut.
#[derive(Clone, Debug)]
pub struct CovalentCut {
    pub bda: usize,
    pub baa: usize,
    pub image: [i32; 3],
}

#[derive(Clone, Debug)]
pub struct FragmentPartition {
    /// Disjoint, nonempty atom lists covering the reference cell exactly once.
    pub atoms: Vec<Vec<usize>>,
    /// Fixed fragment charges, summing to the system's total charge.
    pub charges: Vec<f64>,
    pub cuts: Vec<CovalentCut>,
}
#[derive(Clone, Debug)]
pub struct FmoOptions {
    pub order: usize,
    pub max_embedding_iterations: usize,
    pub embedding_tolerance: f64,
    /// Finite positive HOP constraint in Hartree, as in the reference method.
    pub hop_penalty: f64,
    pub compute_derivatives: bool,
    pub max_ao: usize,
    pub max_clusters: usize,
}
impl Default for FmoOptions {
    fn default() -> Self {
        Self {
            order: 2,
            max_embedding_iterations: 200,
            embedding_tolerance: 1e-9,
            hop_penalty: 1e6,
            compute_derivatives: true,
            max_ao: 512,
            max_clusters: 2048,
        }
    }
}
#[derive(Clone, Debug)]
pub struct FragmentResult {
    pub total_free: f64,
    pub gradient: Vec<Vec3>,
    pub forces: Vec<Vec3>,
    /// `sigma_ab=(1/V)dF/d epsilon_ab` at fixed fractional coordinates.
    pub stress: Option<Matrix>,
    pub shell_charges: Vec<f64>,
    pub embedding_iterations: usize,
    pub local_scc_iterations: usize,
    pub cluster_count: usize,
    pub largest_cluster_ao: usize,
}

fn invalid(message: &str) -> Gfn1Error {
    Gfn1Error::InvalidInput(message.into())
}
fn validate(
    system: &PeriodicSystem,
    options: &ElectronicOptions,
    partition: &FragmentPartition,
) -> Result<()> {
    let nat = system.atoms.len();
    if nat == 0 || partition.atoms.is_empty() || partition.atoms.len() != partition.charges.len() {
        return Err(invalid(
            "fragment partition and charge lengths must match and be nonempty",
        ));
    }
    let mut owner = vec![usize::MAX; nat];
    for (i, atoms) in partition.atoms.iter().enumerate() {
        if atoms.is_empty() {
            return Err(invalid("empty fragment"));
        }
        for &a in atoms {
            if a >= nat || owner[a] != usize::MAX {
                return Err(invalid(
                    "fragment atoms must cover the system exactly once without duplicates",
                ));
            }
            owner[a] = i;
        }
    }
    if owner.contains(&usize::MAX)
        || partition.charges.iter().any(|q| !q.is_finite())
        || (partition.charges.iter().sum::<f64>() - options.charge.unwrap_or(system.charge)).abs()
            > 1e-9
    {
        return Err(invalid("fragment charges must be finite and sum to the system charge, with every atom assigned"));
    }
    let mut detached = vec![false; nat];
    for cut in &partition.cuts {
        if cut.bda >= nat
            || cut.baa >= nat
            || cut.bda == cut.baa
            || owner[cut.bda] == owner[cut.baa]
            || detached[cut.bda]
        {
            return Err(invalid(
                "HOP cuts require different fragments and at most one detached bond per BDA",
            ));
        }
        if system.lattice.is_none() && cut.image != [0; 3] {
            return Err(invalid("molecular cut images must be zero"));
        }
        detached[cut.bda] = true;
    }
    if options.solvation.is_some()
        || options.ml_model.is_some()
        || options.multipole
        || options.lr_exchange
        || options.onsite_exchange
        || options.experimental_d4
        || options.spin_polarization
        || options.plus_u
        || options.charge_order != 3
        || options.external_field.is_active()
        || options.spin_multiplicity.is_some_and(|m| m != 1)
    {
        return Err(invalid("fragment methods currently implement the restricted isotropic third-order GFN1 model with D3; other model modifiers are unsupported"));
    }
    if !options.electronic_temperature.is_finite()
        || options.electronic_temperature < 0.
        || options.max_scc == 0
        || !(options.charge_tolerance.is_finite() && options.charge_tolerance > 0.)
    {
        return Err(invalid("fragment temperature and SCC controls are invalid"));
    }
    if let Some(lattice) = system.lattice {
        if lattice.periodic != [true; 3] {
            return Err(invalid(
                "periodic fragment electrostatics currently require three-dimensional periodicity",
            ));
        }
    }
    Ok(())
}

fn aggregate(clusters: &[Cluster], states: &[LocalState], nsh: usize) -> Vec<f64> {
    let mut q = vec![0.; nsh];
    for (cluster, state) in clusters.iter().zip(states) {
        for (i, &s) in cluster.shells.iter().enumerate() {
            q[s] += state.q[i];
        }
    }
    q
}
fn outside(
    total: &[f64],
    monomers: &[Cluster],
    states: &[LocalState],
    members: &[usize],
) -> Vec<f64> {
    let mut env = total.to_vec();
    for &i in members {
        for (k, &s) in monomers[i].shells.iter().enumerate() {
            env[s] -= states[i].q[k];
        }
    }
    env
}
struct Subsystem {
    members: Vec<usize>,
    coefficient: f64,
    cluster: Cluster,
    state: LocalState,
}
fn cluster_coefficient(n: usize, k: usize, order: usize) -> f64 {
    if order >= n {
        return if k == n { 1. } else { 0. };
    }
    let choose =
        |a: usize, b: usize| -> f64 { (0..b).fold(1., |v, i| v * (a - i) as f64 / (i + 1) as f64) };
    (if (order - k) % 2 == 0 { 1. } else { -1. }) * choose(n - k - 1, order - k)
}

/// FMO2/FMO3 with self-consistent electrostatic monomer embedding and the
/// charge-transfer embedding energy correction. Periodic mode is the coherent
/// Gamma reference-cell extension, using image-summed AO blocks and Ewald
/// electrostatics; it does not approximate an arbitrary k-point mesh.
pub fn fmo_xtb(
    system: &PeriodicSystem,
    params: &Gfn1Parameters,
    electronic: &ElectronicOptions,
    pbc: &PbcOptions,
    partition: &FragmentPartition,
    options: &FmoOptions,
) -> Result<FragmentResult> {
    validate(system, electronic, partition)?;
    if !matches!(options.order, 2 | 3)
        || options.max_embedding_iterations == 0
        || !(options.embedding_tolerance.is_finite() && options.embedding_tolerance > 0.)
        || !(options.hop_penalty.is_finite() && options.hop_penalty > 0.)
    {
        return Err(invalid(
            "invalid FMO order, convergence tolerance or HOP penalty",
        ));
    }
    if !pbc.kmesh.is_gamma_only() && system.lattice.is_some() {
        return Err(invalid("periodic FMO currently uses Gamma reference-cell fragment orbitals; use a supercell for k-space convergence"));
    }
    let basis = crate::basis::BasisSet::build(
        system,
        params,
        crate::basis::BasisOptions {
            nprim: electronic.nprim,
        },
    )?;
    if basis.len() > options.max_ao {
        return Err(invalid("FMO reference-cell AO count exceeds max_ao"));
    }
    let geometry = Geometry::build(system, params, electronic, pbc)?;
    let nat = system.atoms.len();
    let nsh = geometry.gamma.rows();
    let nf = partition.atoms.len();
    let monomers: Vec<_> = partition
        .atoms
        .iter()
        .enumerate()
        .map(|(i, a)| {
            Cluster::new(
                &geometry,
                system,
                a,
                a,
                partition.charges[i],
                &partition.cuts,
                options.hop_penalty,
            )
        })
        .collect::<Result<_>>()?;
    let offsets: Vec<_> = std::iter::once(0)
        .chain(monomers.iter().scan(0, |sum, c| {
            *sum += c.shells.len();
            Some(*sum)
        }))
        .collect();
    let dim = offsets[nf];
    let mut guess = vec![0.; dim];
    let mut mixer = BroydenMixer::new(dim, 20, 0.25);
    let mut embedding_iterations = 0;
    let mut states = Vec::new();
    let mut last_rms = f64::INFINITY;
    let mut local_iterations = 0;
    for iteration in 1..=options.max_embedding_iterations {
        let mut total = vec![0.; nsh];
        for i in 0..nf {
            for (j, &s) in monomers[i].shells.iter().enumerate() {
                total[s] += guess[offsets[i] + j];
            }
        }
        states.clear();
        for i in 0..nf {
            let mut env = total.clone();
            for (j, &s) in monomers[i].shells.iter().enumerate() {
                env[s] -= guess[offsets[i] + j];
            }
            let state = cluster::solve(
                &monomers[i],
                &matrix_vector_product(&geometry.gamma, &env)?,
                electronic,
            )?;
            local_iterations += state.iterations;
            states.push(state);
        }
        let new: Vec<_> = states.iter().flat_map(|s| s.q.iter().copied()).collect();
        let residual: Vec<_> = new.iter().zip(&guess).map(|(a, b)| a - b).collect();
        last_rms = (residual.iter().map(|x| x * x).sum::<f64>() / dim as f64).sqrt();
        embedding_iterations = iteration;
        if last_rms < options.embedding_tolerance {
            break;
        }
        guess = mixer
            .next(&guess, &residual)
            .filter(|q| q.iter().all(|x| x.is_finite() && x.abs() < 20.))
            .unwrap_or_else(|| {
                guess
                    .iter()
                    .zip(&residual)
                    .map(|(a, b)| a + 0.2 * b)
                    .collect()
            });
    }
    if last_rms >= options.embedding_tolerance {
        return Err(Gfn1Error::SccNotConverged {
            iterations: embedding_iterations,
            rms: last_rms,
        });
    }
    let total = aggregate(&monomers, &states, nsh);
    let mut subsystems = Vec::new();
    let order = options.order.min(nf);
    for i in 0..nf {
        for j in i + 1..nf {
            for k in j..nf {
                let members = if k == j { vec![i, j] } else { vec![i, j, k] };
                if members.len() > order {
                    continue;
                }
                let coefficient = cluster_coefficient(nf, members.len(), order);
                if coefficient == 0. {
                    continue;
                }
                if subsystems.len() + nf >= options.max_clusters {
                    return Err(invalid("FMO subsystem count exceeds max_clusters"));
                }
                let atoms: Vec<_> = members
                    .iter()
                    .flat_map(|&f| partition.atoms[f].iter().copied())
                    .collect();
                let charge = members.iter().map(|&f| partition.charges[f]).sum();
                let cluster = Cluster::new(
                    &geometry,
                    system,
                    &atoms,
                    &atoms,
                    charge,
                    &partition.cuts,
                    options.hop_penalty,
                )?;
                let env = outside(&total, &monomers, &states, &members);
                let state = cluster::solve(
                    &cluster,
                    &matrix_vector_product(&geometry.gamma, &env)?,
                    electronic,
                )?;
                local_iterations += state.iterations;
                subsystems.push(Subsystem {
                    members,
                    coefficient,
                    cluster,
                    state,
                });
            }
        }
    }
    let mono_coefficient = cluster_coefficient(nf, 1, order);
    let mut energy = geometry.classical - geometry.repulsion;
    for i in 0..nf {
        energy += mono_coefficient
            * (monomers[i].physical_energy(&states[i].q, &states[i].spec)?
                + geometry.repulsion(&monomers[i].scales, None));
    }
    for sub in &subsystems {
        let env = outside(&total, &monomers, &states, &sub.members);
        let mut transfer = sub.cluster.global_q(&sub.state.q, nsh);
        for &i in &sub.members {
            for (j, &s) in monomers[i].shells.iter().enumerate() {
                transfer[s] -= states[i].q[j];
            }
        }
        let potential = matrix_vector_product(&geometry.gamma, &env)?;
        let correction = transfer
            .iter()
            .zip(&potential)
            .map(|(a, b)| a * b)
            .sum::<f64>();
        energy += sub.coefficient
            * (sub.cluster.physical_energy(&sub.state.q, &sub.state.spec)?
                + geometry.repulsion(&sub.cluster.scales, None)
                + correction);
    }
    let mut derivatives = vec![0.; geometry.ndof];
    if options.compute_derivatives {
        let screened: Vec<_> = monomers
            .iter()
            .zip(&states)
            .map(|(c, s)| Screened::new(c, s))
            .collect::<Result<_>>()?;
        let pair_screened: Vec<_> = subsystems
            .iter()
            .map(|s| Screened::new(&s.cluster, &s.state))
            .collect::<Result<_>>()?;
        let mut jacobian = Matrix::identity(dim);
        for other in 0..nf {
            for (j, &shell) in monomers[other].shells.iter().enumerate() {
                let column = offsets[other] + j;
                for i in 0..nf {
                    if i == other {
                        continue;
                    }
                    let potential: Vec<_> = monomers[i]
                        .shells
                        .iter()
                        .map(|&s| geometry.gamma[(s, shell)])
                        .collect();
                    let zero = Matrix::zeros(monomers[i].aos.len(), monomers[i].aos.len());
                    let (dq, _, _) = screened[i]
                        .response(&monomers[i].shift(&potential, &monomers[i].s), &zero)?;
                    for (k, &value) in dq.iter().enumerate() {
                        jacobian[(offsets[i] + k, column)] -= value;
                    }
                }
            }
        }
        let lu = DenseLu::factor(&jacobian)?;
        for (y, result) in derivatives.iter_mut().enumerate() {
            let direction = geometry.direction(y);
            let explicit = matrix_vector_product(&direction.gamma, &total)?;
            let mut rhs = vec![0.; dim];
            for i in 0..nf {
                let dv: Vec<_> = monomers[i].shells.iter().map(|&s| explicit[s]).collect();
                let (dh, ds) = monomers[i].fixed_fock_derivative(
                    &direction,
                    &states[i].potential,
                    &dv,
                    y,
                    3 * nat,
                    options.hop_penalty,
                )?;
                let (dq, _, _) = screened[i].response(&dh, &ds)?;
                rhs[offsets[i]..offsets[i + 1]].copy_from_slice(&dq);
            }
            let dq_flat = lu.solve_vec(&rhs)?;
            let mut dq_total = vec![0.; nsh];
            for i in 0..nf {
                for (j, &s) in monomers[i].shells.iter().enumerate() {
                    dq_total[s] += dq_flat[offsets[i] + j];
                }
            }
            let all = vec![1.; nat];
            let mut derivative = direction.classical - geometry.repulsion(&all, Some(y));
            // Reconstruct each monomer response with its moving environment.
            for i in 0..nf {
                let dq = &dq_flat[offsets[i]..offsets[i + 1]];
                let mut moving = dq_total.clone();
                for (j, &s) in monomers[i].shells.iter().enumerate() {
                    moving[s] -= dq[j];
                }
                let dv_global = matrix_vector_product(&geometry.gamma, &moving)?;
                let dv: Vec<_> = monomers[i]
                    .shells
                    .iter()
                    .map(|&s| explicit[s] + dv_global[s])
                    .collect();
                let (dh, ds) = monomers[i].fixed_fock_derivative(
                    &direction,
                    &states[i].potential,
                    &dv,
                    y,
                    3 * nat,
                    options.hop_penalty,
                )?;
                let (_, dp, de) = screened[i].response(&dh, &ds)?;
                derivative += mono_coefficient
                    * (monomers[i].physical_derivative(
                        &states[i].q,
                        dq,
                        &states[i].spec,
                        &dp,
                        &de,
                        &direction,
                    )? + geometry.repulsion(&monomers[i].scales, Some(y)));
            }
            for (index, sub) in subsystems.iter().enumerate() {
                let env = outside(&total, &monomers, &states, &sub.members);
                let mut d_env = dq_total.clone();
                let mut transfer = sub.cluster.global_q(&sub.state.q, nsh);
                let mut d_transfer = vec![0.; nsh];
                for &i in &sub.members {
                    for (j, &s) in monomers[i].shells.iter().enumerate() {
                        transfer[s] -= states[i].q[j];
                        d_env[s] -= dq_flat[offsets[i] + j];
                        d_transfer[s] -= dq_flat[offsets[i] + j];
                    }
                }
                let own = sub.cluster.global_q(&sub.state.q, nsh);
                let field: Vec<_> = env.iter().zip(&own).map(|(a, b)| a + b).collect();
                let mut dv = matrix_vector_product(&direction.gamma, &field)?;
                let dynamic = matrix_vector_product(&geometry.gamma, &d_env)?;
                for (a, b) in dv.iter_mut().zip(dynamic) {
                    *a += b;
                }
                let dv: Vec<_> = sub.cluster.shells.iter().map(|&s| dv[s]).collect();
                let (dh, ds) = sub.cluster.fixed_fock_derivative(
                    &direction,
                    &sub.state.potential,
                    &dv,
                    y,
                    3 * nat,
                    options.hop_penalty,
                )?;
                let (dq, dp, de) = pair_screened[index].response(&dh, &ds)?;
                for (j, &s) in sub.cluster.shells.iter().enumerate() {
                    d_transfer[s] += dq[j];
                }
                let v = matrix_vector_product(&geometry.gamma, &env)?;
                let mut dv = matrix_vector_product(&direction.gamma, &env)?;
                let dynamic = matrix_vector_product(&geometry.gamma, &d_env)?;
                for (a, b) in dv.iter_mut().zip(dynamic) {
                    *a += b;
                }
                let correction = d_transfer.iter().zip(&v).map(|(a, b)| a * b).sum::<f64>()
                    + transfer.iter().zip(&dv).map(|(a, b)| a * b).sum::<f64>();
                derivative += sub.coefficient
                    * (sub.cluster.physical_derivative(
                        &sub.state.q,
                        &dq,
                        &sub.state.spec,
                        &dp,
                        &de,
                        &direction,
                    )? + geometry.repulsion(&sub.cluster.scales, Some(y))
                        + correction);
            }
            *result = derivative;
        }
    }
    let largest = monomers
        .iter()
        .map(|c| c.aos.len())
        .chain(subsystems.iter().map(|s| s.cluster.aos.len()))
        .max()
        .unwrap_or(0);
    if !options.compute_derivatives {
        derivatives.clear();
    }
    finish(
        energy,
        derivatives,
        total,
        embedding_iterations,
        local_iterations,
        nf + subsystems.len(),
        largest,
        &geometry,
    )
}

#[allow(clippy::too_many_arguments)]
fn finish(
    energy: f64,
    derivatives: Vec<f64>,
    charges: Vec<f64>,
    iterations: usize,
    local_iterations: usize,
    clusters: usize,
    largest: usize,
    geometry: &Geometry,
) -> Result<FragmentResult> {
    if !energy.is_finite() || derivatives.iter().any(|x| !x.is_finite()) {
        return Err(invalid("non-finite fragment energy or derivative"));
    }
    let requested = !derivatives.is_empty();
    let ncart = if requested {
        geometry.ndof - if geometry.volume.is_some() { 9 } else { 0 }
    } else {
        0
    };
    let gradient: Vec<_> = derivatives[..ncart]
        .chunks_exact(3)
        .map(|d| Vec3::new(d[0], d[1], d[2]))
        .collect();
    let forces = gradient.iter().map(|g| -*g).collect();
    let stress = geometry.volume.filter(|_| requested).map(|v| {
        Matrix::from_vec(3, 3, derivatives[ncart..].iter().map(|d| d / v).collect())
            .expect("nine strains")
    });
    Ok(FragmentResult {
        total_free: energy,
        gradient,
        forces,
        stress,
        shell_charges: charges,
        embedding_iterations: iterations,
        local_scc_iterations: local_iterations,
        cluster_count: clusters,
        largest_cluster_ao: largest,
    })
}
