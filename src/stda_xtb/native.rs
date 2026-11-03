// SPDX-License-Identifier: GPL-3.0-or-later
//! Native simplified response on dedicated sTDA-xTB orbitals.
use super::ground::{ground_state_at_temperature, tables, EV_PER_HARTREE};
use crate::error::{Gfn1Error, Result};
use crate::integrals::contracted_pair;
use crate::linalg::{lowdin_orthogonalizer, matrix_vector_product, symmetric_eigen, Matrix};
use crate::system::PeriodicSystem;

#[derive(Clone, Debug)]
pub struct NativeStdaOptions {
    /// Kelvin, used by both VTB updates and the final occupation check.
    pub electronic_temperature: f64,
    /// Select the finite-temperature ensemble CI. None selects it for fractional final occupations or orbital degeneracy.
    pub thermal_response: Option<bool>,
    /// Differentiate the mean of the CI manifold containing the selected root.
    pub manifold_average: bool,
    pub energy_window_ev: f64,
    pub n_states: usize,
    pub triplet: bool,
    /// Perturbative configuration selection threshold in Hartree.
    pub pt_threshold: f64,
    pub max_ao: usize,
    pub max_configurations: usize,
    pub memory_budget_bytes: usize,
}
impl Default for NativeStdaOptions {
    fn default() -> Self {
        Self {
            electronic_temperature: 300.,
            thermal_response: None,
            manifold_average: false,
            energy_window_ev: 10.,
            n_states: 10,
            triplet: false,
            pt_threshold: 1e-4,
            max_ao: 512,
            max_configurations: 4096,
            memory_budget_bytes: 536870912,
        }
    }
}
#[derive(Clone, Debug)]
pub struct NativeStdaState {
    pub excitation_hartree: f64,
    pub excitation_ev: f64,
    pub oscillator_strength: f64,
    pub transition_dipole: [f64; 3],
    /// Amplitudes in the returned selected occupied/virtual configuration list.
    pub amplitudes: Vec<f64>,
}
#[derive(Clone, Debug)]
pub struct NativeStdaResult {
    pub thermal_response: bool,
    pub electronic_temperature_kelvin: f64,
    pub maximum_occupation_deviation: f64,
    /// Lowest eigenvalue before positive-root spectral filtering.
    pub lowest_ci_eigenvalue_hartree: f64,
    pub nonpositive_ci_roots: usize,
    /// Complete signed CI eigenvalues, before the positive absorption filter.
    pub ci_eigenvalues_hartree: Vec<f64>,
    pub states: Vec<NativeStdaState>,
    pub configurations: Vec<(usize, usize)>,
    pub primary_configuration_count: usize,
    pub promoted_configuration_count: usize,
    pub orbital_energies_hartree: Vec<f64>,
    pub vtb_charges: Vec<f64>,
    pub basis_size: usize,
}
fn invalid(message: impl Into<String>) -> Gfn1Error {
    Gfn1Error::InvalidInput(message.into())
}
fn pair(i: usize, j: usize) -> usize {
    let (a, b) = if i >= j { (i, j) } else { (j, i) };
    a * (a + 1) / 2 + b
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
struct Candidate {
    i: usize,
    a: usize,
    diagonal: f64,
}

/// Pure-Rust sTDA-xTB: dedicated two-stage orbital generator, Lowdin transition
/// charges, Mataga-Nishimoto/Ohno response kernels, upstream -xtb Kia shift,
/// perturbative configuration selection and AO-integral length-gauge strengths.
/// This does not call an external executable or the TD-GFN1 TDA implementation.
pub fn solve_stda_xtb_native(
    system: &PeriodicSystem,
    options: &NativeStdaOptions,
) -> Result<NativeStdaResult> {
    solve_native(system, options, false)
}

/// Keep the complete CI eigensystem for analytic eigenvector response, without
/// changing the energy window used to select the configuration space.
pub(super) fn solve_stda_xtb_native_response(
    system: &PeriodicSystem,
    options: &NativeStdaOptions,
) -> Result<NativeStdaResult> {
    solve_native(system, options, true)
}

fn solve_native(
    system: &PeriodicSystem,
    options: &NativeStdaOptions,
    retain_all_ci_roots: bool,
) -> Result<NativeStdaResult> {
    let options = &NativeStdaOptions {
        memory_budget_bytes: crate::resources::effective_memory_budget(options.memory_budget_bytes),
        ..options.clone()
    };
    if !(options.energy_window_ev.is_finite() && options.energy_window_ev > 0.)
        || !(options.pt_threshold.is_finite() && options.pt_threshold > 0.)
        || options.n_states == 0
        || options.max_configurations == 0
    {
        return Err(invalid(
            "invalid native sTDA-xTB energy window, state count or configuration controls",
        ));
    }
    let bounded_ao = options
        .max_ao
        .min((options.memory_budget_bytes / 128).isqrt());
    let orbital = ground_state_at_temperature(system, bounded_ao, options.electronic_temperature)?;
    if super::thermal::use_ensemble(&orbital, options)? {
        return super::thermal::spectrum(system, &orbital, options, retain_all_ci_roots);
    }
    let nat = system.atoms.len();
    let nao = orbital.aos.len();
    let nocc = orbital.nocc;
    let threshold = options.energy_window_ev / 27.211385050;
    let range = 2. * (1. + 0.8 * 0.5) * threshold;
    let occupied: Vec<_> = (0..nocc)
        .filter(|&i| orbital.energies[i] > orbital.energies[nocc] - range)
        .collect();
    let virtuals: Vec<_> = (nocc..nao)
        .filter(|&a| {
            orbital.energies[a] <= 3. && orbital.energies[a] < orbital.energies[nocc - 1] + range
        })
        .collect();
    if occupied.is_empty() || virtuals.is_empty() {
        return Err(invalid(
            "no sTDA-xTB configurations in this orbital energy window",
        ));
    }
    let orbitals: Vec<_> = occupied.iter().chain(&virtuals).copied().collect();
    let nmo = orbitals.len();
    let npair = nmo
        .checked_mul(nmo + 1)
        .and_then(|x| x.checked_div(2))
        .ok_or_else(|| invalid("sTDA pair dimension overflow"))?;
    let charge_memory = npair
        .checked_mul(nat)
        .and_then(|x| x.checked_mul(24))
        .ok_or_else(|| invalid("sTDA charge-cache allocation overflow"))?;
    if charge_memory > options.memory_budget_bytes / 2 {
        return Err(invalid("native sTDA-xTB transition-charge cache exceeds memory_budget_bytes; reduce the orbital energy window"));
    }
    let orth = lowdin_orthogonalizer(&orbital.overlap, 1e-10)?.x;
    let clow = orbital
        .overlap
        .matmul(&orth)?
        .matmul(&orbital.coefficients)?;
    let mut q = vec![vec![0.; nat]; npair];
    for i in 0..nmo {
        for j in 0..=i {
            let values = &mut q[pair(i, j)];
            for mu in 0..nao {
                values[orbital.aos[mu].atom_index] +=
                    clow[(mu, orbitals[i])] * clow[(mu, orbitals[j])];
            }
        }
    }
    // This hardness is the response model's setrep table, not either fitted
    // Hamiltonian's GAM overrides.
    let mut gamma_j = Matrix::zeros(nat, nat);
    let mut gamma_k = Matrix::zeros(nat, nat);
    for i in 0..nat {
        for j in 0..=i {
            let eta = 0.5
                * (tables()["default_hardness"][usize::from(system.atoms[i].z) - 1]
                    .as_f64()
                    .expect("hardness")
                    + tables()["default_hardness"][usize::from(system.atoms[j].z) - 1]
                        .as_f64()
                        .expect("hardness"));
            let distance = (system.atoms[i].position - system.atoms[j].position).norm();
            let gj = (distance.powi(4) + (0.5 * eta).powi(-4)).powf(-0.25);
            let gk = (distance.powi(2) + eta.powi(-2)).powf(-0.5);
            gamma_j[(i, j)] = gj;
            gamma_j[(j, i)] = gj;
            gamma_k[(i, j)] = gk;
            gamma_k[(j, i)] = gk;
        }
    }
    let jq = q
        .iter()
        .map(|q| matrix_vector_product(&gamma_j, q))
        .collect::<Result<Vec<_>>>()?;
    let kq = q
        .iter()
        .map(|q| matrix_vector_product(&gamma_k, q))
        .collect::<Result<Vec<_>>>()?;
    let ak = if options.triplet { 0. } else { 2. };
    let coupling = |i: usize, a: usize, j: usize, b: usize| {
        ak * dot(&q[pair(i, a)], &kq[pair(j, b)]) - dot(&q[pair(i, j)], &jq[pair(a, b)])
    };
    let mut primary = Vec::new();
    let mut secondary = Vec::new();
    for i in 0..occupied.len() {
        for v in 0..virtuals.len() {
            let a = occupied.len() + v;
            let ek = dot(&q[pair(i, a)], &kq[pair(i, a)]);
            let kia_shift = (0.5 * 0.036749320) / (1. + (ek / (0.1 * 0.036749320)).powi(4));
            let diagonal = orbital.energies[orbitals[a]] - orbital.energies[orbitals[i]]
                + coupling(i, a, i, a)
                + kia_shift;
            let candidate = Candidate { i, a, diagonal };
            if diagonal <= threshold {
                primary.push(candidate);
            } else {
                secondary.push(candidate);
            }
        }
    }
    if primary.is_empty() {
        return Err(invalid(
            "no primary sTDA-xTB configurations; increase energy_window_ev",
        ));
    }
    primary.sort_by(|a, b| a.diagonal.total_cmp(&b.diagonal));
    let primary_count = primary.len();
    let mut perturbation = vec![0.; primary_count];
    let mut promoted = Vec::new();
    for secondary in secondary {
        let contributions: Vec<_> = primary
            .iter()
            .map(|p| {
                coupling(secondary.i, secondary.a, p.i, p.a).powi(2)
                    / (secondary.diagonal - p.diagonal + 1e-10)
            })
            .collect();
        if contributions.iter().sum::<f64>() > options.pt_threshold {
            promoted.push(secondary);
        } else {
            for (p, c) in perturbation.iter_mut().zip(contributions) {
                *p += c;
            }
        }
    }
    for (p, shift) in primary.iter_mut().zip(perturbation) {
        p.diagonal -= shift;
    }
    let promoted_count = promoted.len();
    primary.extend(promoted);
    let nci = primary.len();
    if nci > options.max_configurations {
        return Err(invalid(format!(
            "native sTDA-xTB selected {nci} configurations, exceeding max_configurations"
        )));
    }
    let matrix_memory = nci
        .checked_mul(nci)
        .and_then(|x| x.checked_mul(48))
        .ok_or_else(|| invalid("sTDA matrix allocation overflow"))?;
    if matrix_memory > options.memory_budget_bytes.saturating_sub(charge_memory) {
        return Err(invalid(
            "native sTDA-xTB CI solve exceeds memory_budget_bytes",
        ));
    }
    let mut matrix = Matrix::zeros(nci, nci);
    for i in 0..nci {
        matrix[(i, i)] = primary[i].diagonal;
        for j in 0..i {
            let value = coupling(primary[i].i, primary[i].a, primary[j].i, primary[j].a);
            matrix[(i, j)] = value;
            matrix[(j, i)] = value;
        }
    }
    let eigen = symmetric_eigen(&matrix)?;
    let mut ao_dipole = [
        Matrix::zeros(nao, nao),
        Matrix::zeros(nao, nao),
        Matrix::zeros(nao, nao),
    ];
    for i in 0..nao {
        for j in 0..=i {
            let pair = contracted_pair(
                &orbital.aos[i],
                &orbital.aos[j],
                system.atoms[orbital.aos[i].atom_index].position,
                system.atoms[orbital.aos[j].atom_index].position,
            );
            for (axis, value) in [pair.1, pair.2, pair.3].into_iter().enumerate() {
                // The shared multipole builder is ket-centred. Translate to
                // the laboratory origin before imposing Hermitian symmetry.
                let absolute = value
                    + system.atoms[orbital.aos[j].atom_index].position.to_array()[axis] * pair.0;
                ao_dipole[axis][(i, j)] = -absolute;
                ao_dipole[axis][(j, i)] = -absolute;
            }
        }
    }
    let mo_dipole = ao_dipole
        .iter()
        .map(|d| {
            orbital
                .coefficients
                .transpose()
                .matmul(d)?
                .matmul(&orbital.coefficients)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut states = Vec::new();
    for (root, &energy) in eigen.values.iter().enumerate() {
        if !retain_all_ci_roots && (energy <= 0. || energy > threshold) {
            continue;
        }
        if !retain_all_ci_roots && states.len() >= options.n_states {
            break;
        }
        let mut dipole = [0.; 3];
        let amplitudes = eigen.vectors.column(root);
        for (candidate, &amplitude) in primary.iter().zip(&amplitudes) {
            for axis in 0..3 {
                dipole[axis] += ak.sqrt()
                    * amplitude
                    * mo_dipole[axis][(orbitals[candidate.i], orbitals[candidate.a])];
            }
        }
        states.push(NativeStdaState {
            excitation_hartree: energy,
            excitation_ev: energy * EV_PER_HARTREE,
            oscillator_strength: (2. / 3.)
                * energy.max(0.)
                * dipole.iter().map(|d| d * d).sum::<f64>(),
            transition_dipole: dipole,
            amplitudes,
        });
    }
    if states.is_empty() {
        return Err(invalid(
            "no positive native sTDA-xTB excitation in the requested window",
        ));
    }
    Ok(NativeStdaResult {
        thermal_response: false,
        electronic_temperature_kelvin: orbital.electronic_temperature_kelvin,
        maximum_occupation_deviation: orbital.maximum_occupation_deviation,
        lowest_ci_eigenvalue_hartree: eigen.values[0],
        nonpositive_ci_roots: eigen.values.iter().filter(|&&value| value <= 0.).count(),
        ci_eigenvalues_hartree: eigen.values,
        states,
        configurations: primary
            .iter()
            .map(|p| (orbitals[p.i], orbitals[p.a]))
            .collect(),
        primary_configuration_count: primary_count,
        promoted_configuration_count: promoted_count,
        orbital_energies_hartree: orbital.energies,
        vtb_charges: orbital.vtb_charges,
        basis_size: nao,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_stda_spectra_match_independent_std2() {
        let reference: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/data/stda-spectrum-reference.json"
        ))
        .unwrap();
        for case in reference["cases"].as_array().unwrap() {
            let atoms = case["numbers"]
                .as_array()
                .unwrap()
                .iter()
                .zip(case["positions_bohr"].as_array().unwrap())
                .map(|(z, r)| crate::system::Atom {
                    z: z.as_u64().unwrap() as u8,
                    position: crate::math::Vec3::new(
                        r[0].as_f64().unwrap(),
                        r[1].as_f64().unwrap(),
                        r[2].as_f64().unwrap(),
                    ),
                })
                .collect();
            let result = solve_stda_xtb_native(
                &PeriodicSystem::new(atoms, None),
                &NativeStdaOptions {
                    energy_window_ev: 20.,
                    n_states: 100,
                    ..Default::default()
                },
            )
            .unwrap();
            let expected = case["states"].as_array().unwrap();
            assert_eq!(result.states.len(), expected.len());
            let mut e = 0.0_f64;
            let mut f = 0.0_f64;
            for (a, b) in result.states.iter().zip(expected) {
                e = e.max((a.excitation_ev - b["excitation_ev"].as_f64().unwrap()).abs());
                f = f.max(
                    (a.oscillator_strength - b["oscillator_strength"].as_f64().unwrap()).abs(),
                );
            }
            eprintln!(
                "{} native sTDA all {} states energy/fL max errors {e:.3e}/{f:.3e}",
                case["name"],
                result.states.len()
            );
            assert!(
                e < 2e-4 && f < 1e-4,
                "native sTDA vs independent official response"
            );
            // std2 uses single-precision charge contractions and CI algebra.
            // Two formaldehyde roots separated by 0.00082 eV exchange about
            // 7e-5 of fL under that rounding. Their summed strength is stable.
            let mut start = 0;
            while start < result.states.len() {
                let mut end = start + 1;
                while end < result.states.len()
                    && result.states[end].excitation_ev - result.states[end - 1].excitation_ev
                        < 1e-3
                {
                    end += 1;
                }
                let actual: f64 = result.states[start..end]
                    .iter()
                    .map(|s| s.oscillator_strength)
                    .sum();
                let expected: f64 = expected[start..end]
                    .iter()
                    .map(|s| s["oscillator_strength"].as_f64().unwrap())
                    .sum();
                assert!(
                    (actual - expected).abs() < 2e-5,
                    "isolated-root or near-degenerate-subspace strength mismatch"
                );
                start = end;
            }
        }
    }
}
