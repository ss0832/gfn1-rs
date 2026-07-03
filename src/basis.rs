// SPDX-License-Identifier: GPL-3.0-or-later
use crate::error::{Gfn1Error, Result};
use crate::params::{AngularMomentum, Gfn1Parameters};
use crate::sto::{slater_to_gauss, PrimitiveGaussian};
use crate::system::PeriodicSystem;

// GFN1-xTB was parametrized with the older eV->Eh conversion (1/27.21138505),
// not the CODATA value. xtb/tblite both use this older constant deliberately
// ("for consistency"); the GFN1 self-energies and CN shifts were fit against it,
// so reproducing GFN1 requires the same factor (a ~4.4e-8 relative offset
// otherwise propagates into every H0 diagonal and CN shift).
pub const EV_TO_HARTREE: f64 = 1.0 / 27.211_385_05;

/// `nprim = 0` means: use the standard GFN1-xTB primitive-count rule.
pub const AUTO_NPRIM: usize = 0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CartesianPower {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

impl CartesianPower {
    pub const fn new(x: usize, y: usize, z: usize) -> Self {
        Self { x, y, z }
    }
    pub fn total(self) -> usize {
        self.x + self.y + self.z
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CartesianComponent {
    pub power: CartesianPower,
    pub coefficient: f64,
}

impl CartesianComponent {
    pub const fn new(power: CartesianPower, coefficient: f64) -> Self {
        Self { power, coefficient }
    }
}

#[derive(Clone, Debug)]
pub struct AOBasisFunction {
    pub atom_index: usize,
    pub z: u8,
    pub shell_index: usize,
    pub shell_param_index: usize,
    pub shell_label: String,
    pub angular: AngularMomentum,
    pub cart_label: &'static str,
    pub components: Vec<CartesianComponent>,
    pub hdiag_ev: f64,
    pub hdiag_ha: f64,
    pub slater: f64,
    pub principal_n: u8,
    pub nprim: usize,
    pub reference_occ: f64,
    /// Structural valence flag: true iff this is the first shell of its angular
    /// momentum on its atom (xtb `generateValenceShellData` rule). Used for the
    /// H0 shell-pair scaling (valence vs polarization branch); NOT the same as
    /// `reference_occ > 0` (e.g. main-group empty d-shells are valence here).
    pub is_valence: bool,
    pub poly_raw: Option<f64>,
    pub kcn_raw: Option<f64>,
    pub lpar_raw: Option<f64>,
    pub primitives: Vec<PrimitiveGaussian>,
}

#[derive(Clone, Debug)]
pub struct BasisShell {
    pub atom_index: usize,
    pub z: u8,
    pub shell_param_index: usize,
    pub first_ao: usize,
    pub nao: usize,
    pub label: String,
    pub angular: AngularMomentum,
    pub hdiag_ev: f64,
    pub hdiag_ha: f64,
    pub slater: f64,
    pub principal_n: u8,
    pub nprim: usize,
    pub reference_occ: f64,
    /// Structural valence flag (first shell of its angular momentum on the atom,
    /// xtb `generateValenceShellData` rule) for the H0 shell-pair scaling.
    pub is_valence: bool,
    pub poly_raw: Option<f64>,
    pub kcn_raw: Option<f64>,
    pub lpar_raw: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
pub struct BasisOptions {
    /// Set to 0 for the standard GFN1 primitive-count rule; set 1..=6 only for testing.
    pub nprim: usize,
}
impl Default for BasisOptions {
    fn default() -> Self {
        Self { nprim: AUTO_NPRIM }
    }
}

#[derive(Clone, Debug)]
pub struct BasisSet {
    pub aos: Vec<AOBasisFunction>,
    pub shells: Vec<BasisShell>,
    pub reference_electrons: Vec<f64>,
    pub total_reference_electrons: f64,
}

impl BasisSet {
    pub fn build(
        system: &PeriodicSystem,
        params: &Gfn1Parameters,
        options: BasisOptions,
    ) -> Result<Self> {
        let mut aos = Vec::new();
        let mut shells_out = Vec::new();
        let mut reference_electrons = Vec::with_capacity(system.atoms.len());
        for (atom_index, atom) in system.atoms.iter().enumerate() {
            let elem = params.element(atom.z)?;
            let mut atom_ref = 0.0;
            // First-occurrence-of-l tracker for the structural valence flag
            // (xtb `generateValenceShellData`): the first shell of each angular
            // momentum on the atom is valence, later same-l shells are not.
            let mut seen_l = [false; 5];
            let mut same_l_reference: [Option<Vec<PrimitiveGaussian>>; 5] =
                [None, None, None, None, None];
            for (shell_param_index, shell) in elem.shells.iter().enumerate() {
                let first_ao = aos.len();
                let l_index = match shell.l {
                    AngularMomentum::S => 0,
                    AngularMomentum::P => 1,
                    AngularMomentum::D => 2,
                    AngularMomentum::F => 3,
                    AngularMomentum::G => 4,
                };
                let is_valence = !seen_l[l_index];
                seen_l[l_index] = true;
                let components = spherical_components(shell.l)?;
                let requested_nprim = if options.nprim == AUTO_NPRIM {
                    gfn1_number_of_primitives(atom.z, shell.principal_n, shell.l)?
                } else {
                    options.nprim
                };
                let mut primitives = slater_to_gauss(
                    requested_nprim,
                    shell.principal_n,
                    shell.l,
                    shell.slater,
                    true,
                )?;
                if let Some(reference) = &same_l_reference[l_index] {
                    orthogonalize_same_center(reference, &mut primitives);
                } else {
                    same_l_reference[l_index] = Some(primitives.clone());
                }
                let nprim = primitives.len();
                let shell_ref = if is_valence {
                    gfn1_reference_occupation(atom.z, shell.l)
                } else {
                    0.0
                };
                let per_ao_ref = if components.is_empty() {
                    0.0
                } else {
                    shell_ref / components.len() as f64
                };
                let shell_index = shells_out.len();
                for (cart_label, component_terms) in &components {
                    aos.push(AOBasisFunction {
                        atom_index,
                        z: atom.z,
                        shell_index,
                        shell_param_index,
                        shell_label: shell.label.clone(),
                        angular: shell.l,
                        cart_label: *cart_label,
                        components: component_terms.clone(),
                        hdiag_ev: shell.level_ev,
                        hdiag_ha: shell.level_ev * EV_TO_HARTREE,
                        slater: shell.slater,
                        principal_n: shell.principal_n,
                        nprim,
                        reference_occ: per_ao_ref,
                        is_valence,
                        poly_raw: Some(shell.poly_raw * 0.01),
                        kcn_raw: Some(gfn1_kcn_shift(params, shell.level_ev, shell.l, atom.z)),
                        lpar_raw: Some(shell.lpar_raw * 0.1),
                        primitives: primitives.clone(),
                    });
                }
                let nao = aos.len() - first_ao;
                shells_out.push(BasisShell {
                    atom_index,
                    z: atom.z,
                    shell_param_index,
                    first_ao,
                    nao,
                    label: shell.label.clone(),
                    angular: shell.l,
                    hdiag_ev: shell.level_ev,
                    hdiag_ha: shell.level_ev * EV_TO_HARTREE,
                    slater: shell.slater,
                    principal_n: shell.principal_n,
                    nprim,
                    reference_occ: shell_ref,
                    is_valence,
                    poly_raw: Some(shell.poly_raw * 0.01),
                    kcn_raw: Some(gfn1_kcn_shift(params, shell.level_ev, shell.l, atom.z)),
                    lpar_raw: Some(shell.lpar_raw * 0.1),
                });
                atom_ref += shell_ref;
            }
            reference_electrons.push(atom_ref);
        }
        let total_reference_electrons = reference_electrons.iter().sum();
        Ok(Self {
            aos,
            shells: shells_out,
            reference_electrons,
            total_reference_electrons,
        })
    }

    pub fn len(&self) -> usize {
        self.aos.len()
    }
    pub fn is_empty(&self) -> bool {
        self.aos.is_empty()
    }

    pub fn ao_reference_occupations(&self) -> Vec<f64> {
        self.aos.iter().map(|ao| ao.reference_occ).collect()
    }

    pub fn shell_of_ao(&self, ao: usize) -> &BasisShell {
        &self.shells[self.aos[ao].shell_index]
    }
}

pub fn gfn1_number_of_primitives(z: u8, principal_n: u8, l: AngularMomentum) -> Result<usize> {
    use AngularMomentum::*;
    let nprim = if z <= 2 {
        match l {
            S => {
                if principal_n == 1 {
                    4
                } else {
                    3
                }
            }
            P => 3,
            D | F => 4,
            G => 0,
        }
    } else {
        match l {
            S | P => 6,
            D | F => 4,
            G => 0,
        }
    };
    if nprim == 0 {
        return Err(Gfn1Error::InvalidInput(format!(
            "GFN1 primitive-count rule has no support for shell n={principal_n} l={l:?} on Z={z}"
        )));
    }
    Ok(nprim)
}

fn gfn1_kcn_shift(params: &Gfn1Parameters, level_ev: f64, l: AngularMomentum, z: u8) -> f64 {
    let Some(kind) = gfn1_kind(z) else {
        return 0.0;
    };
    let factor = match (kind, l) {
        (1, AngularMomentum::S) | (2, AngularMomentum::S) => params.global("cns", 0.6),
        (1, AngularMomentum::P) | (2, AngularMomentum::P) => params.global("cnp", -0.3),
        (1, AngularMomentum::D) => params.global("cnd1", -0.5),
        (2, AngularMomentum::D) => params.global("cnd2", 0.5),
        _ => 0.0,
    };
    -level_ev * EV_TO_HARTREE * factor * 0.01
}

fn gfn1_kind(z: u8) -> Option<u8> {
    match z {
        1 | 2 | 6..=10 | 14..=18 | 32..=36 | 50..=54 | 82..=86 => Some(1),
        21..=24 => Some(2),
        _ => None,
    }
}

/// GFN1 reference occupation by angular shell. This table is part of the model
/// definition and is not present in `param_gfn1-xtb.txt`.
pub fn gfn1_reference_occupation(z: u8, l: AngularMomentum) -> f64 {
    let zi = z as usize;
    if zi >= GFN1_REFERENCE_OCC.len() {
        return 0.0;
    }
    match l {
        AngularMomentum::S => GFN1_REFERENCE_OCC[zi][0],
        AngularMomentum::P => GFN1_REFERENCE_OCC[zi][1],
        AngularMomentum::D => GFN1_REFERENCE_OCC[zi][2],
        AngularMomentum::F | AngularMomentum::G => 0.0,
    }
}

fn orthogonalize_same_center(reference: &[PrimitiveGaussian], shell: &mut Vec<PrimitiveGaussian>) {
    let mut overlap = 0.0;
    for a in reference {
        for b in shell.iter() {
            let eab = a.exponent + b.exponent;
            let kab = (std::f64::consts::PI / eab).sqrt().powi(3);
            overlap += a.coefficient * b.coefficient * kab;
        }
    }
    shell.extend(reference.iter().map(|p| PrimitiveGaussian {
        exponent: p.exponent,
        coefficient: -overlap * p.coefficient,
    }));

    let mut norm = 0.0;
    for a in shell.iter() {
        for b in shell.iter() {
            let eab = a.exponent + b.exponent;
            let kab = (std::f64::consts::PI / eab).sqrt().powi(3);
            norm += a.coefficient * b.coefficient * kab;
        }
    }
    let scale = norm.sqrt().recip();
    for primitive in shell {
        primitive.coefficient *= scale;
    }
}

/// Neutral valence-electron count used as n0 for atom-resolved Mulliken charges.
pub fn neutral_valence_electrons(z: u8) -> f64 {
    let zi = z as usize;
    if zi >= GFN1_REFERENCE_OCC.len() {
        return z as f64;
    }
    GFN1_REFERENCE_OCC[zi].iter().sum()
}

/// Per-atom smallest (most diffuse) primitive Gaussian exponent, indexed by atom
/// (`nat` entries). Used for overlap distance screening: the contracted-Gaussian
/// overlap decays no slower than `exp(-e_a e_b/(e_a+e_b) r^2)`, so an atom pair at
/// `r^2 * e_a e_b > K (e_a + e_b)` has a negligible overlap (and overlap
/// derivatives) and can be skipped before evaluating the integral. `K = 40` keeps
/// the dropped contribution below `exp(-40) ~ 4e-18`.
pub fn atom_min_exponents(basis: &BasisSet, nat: usize) -> Vec<f64> {
    let mut out = vec![f64::INFINITY; nat];
    for ao in &basis.aos {
        for prim in &ao.primitives {
            if prim.exponent < out[ao.atom_index] {
                out[ao.atom_index] = prim.exponent;
            }
        }
    }
    out
}

/// Overlap distance screening predicate: `true` when the atom pair `(a, b)` at
/// squared distance `r2` has a negligible overlap and can be skipped.
#[inline]
pub fn overlap_screened(atom_min_exp: &[f64], a: usize, b: usize, r2: f64) -> bool {
    let ea = atom_min_exp[a];
    let eb = atom_min_exp[b];
    r2 * ea * eb > 40.0 * (ea + eb)
}

fn spherical_components(
    l: AngularMomentum,
) -> Result<Vec<(&'static str, Vec<CartesianComponent>)>> {
    let sqrt3 = 3.0_f64.sqrt();
    let sqrt3_over_2 = sqrt3 / 2.0;
    match l {
        AngularMomentum::S => Ok(vec![(
            "s",
            vec![CartesianComponent::new(CartesianPower::new(0, 0, 0), 1.0)],
        )]),
        AngularMomentum::P => Ok(vec![
            (
                "px",
                vec![CartesianComponent::new(CartesianPower::new(1, 0, 0), 1.0)],
            ),
            (
                "py",
                vec![CartesianComponent::new(CartesianPower::new(0, 1, 0), 1.0)],
            ),
            (
                "pz",
                vec![CartesianComponent::new(CartesianPower::new(0, 0, 1), 1.0)],
            ),
        ]),
        // Six Cartesian d primitives are transformed to the five real spherical
        // d functions; mixed Cartesian terms carry sqrt(3).
        AngularMomentum::D => Ok(vec![
            (
                "dx2-y2",
                vec![
                    CartesianComponent::new(CartesianPower::new(2, 0, 0), sqrt3_over_2),
                    CartesianComponent::new(CartesianPower::new(0, 2, 0), -sqrt3_over_2),
                ],
            ),
            (
                "dz2",
                vec![
                    CartesianComponent::new(CartesianPower::new(2, 0, 0), 0.5),
                    CartesianComponent::new(CartesianPower::new(0, 2, 0), 0.5),
                    CartesianComponent::new(CartesianPower::new(0, 0, 2), -1.0),
                ],
            ),
            (
                "dxy",
                vec![CartesianComponent::new(CartesianPower::new(1, 1, 0), sqrt3)],
            ),
            (
                "dxz",
                vec![CartesianComponent::new(CartesianPower::new(1, 0, 1), sqrt3)],
            ),
            (
                "dyz",
                vec![CartesianComponent::new(CartesianPower::new(0, 1, 1), sqrt3)],
            ),
        ]),
        AngularMomentum::F | AngularMomentum::G => Err(Gfn1Error::InvalidInput(format!(
            "integral engine currently implements s, p, d shells; got {l:?}"
        ))),
    }
}

const GFN1_REFERENCE_OCC: [[f64; 3]; 87] = [
    [0.0, 0.0, 0.0], // Z=0
    [1.0, 0.0, 0.0], // Z=1
    [2.0, 0.0, 0.0], // Z=2
    [1.0, 0.0, 0.0], // Z=3
    [2.0, 0.0, 0.0], // Z=4
    [2.0, 1.0, 0.0], // Z=5
    [2.0, 2.0, 0.0], // Z=6
    [2.0, 3.0, 0.0], // Z=7
    [2.0, 4.0, 0.0], // Z=8
    [2.0, 5.0, 0.0], // Z=9
    [2.0, 6.0, 0.0], // Z=10
    [1.0, 0.0, 0.0], // Z=11
    [2.0, 0.0, 0.0], // Z=12
    [2.0, 1.0, 0.0], // Z=13
    [2.0, 2.0, 0.0], // Z=14
    [2.0, 3.0, 0.0], // Z=15
    [2.0, 4.0, 0.0], // Z=16
    [2.0, 5.0, 0.0], // Z=17
    [2.0, 6.0, 0.0], // Z=18
    [1.0, 0.0, 0.0], // Z=19
    [2.0, 0.0, 0.0], // Z=20
    [2.0, 0.0, 1.0], // Z=21
    [2.0, 0.0, 2.0], // Z=22
    [2.0, 0.0, 3.0], // Z=23
    [2.0, 0.0, 4.0], // Z=24
    [2.0, 0.0, 5.0], // Z=25
    [2.0, 0.0, 6.0], // Z=26
    [2.0, 0.0, 7.0], // Z=27
    [2.0, 0.0, 8.0], // Z=28
    [2.0, 0.0, 9.0], // Z=29
    [2.0, 0.0, 0.0], // Z=30
    [2.0, 1.0, 0.0], // Z=31
    [2.0, 2.0, 0.0], // Z=32
    [2.0, 3.0, 0.0], // Z=33
    [2.0, 4.0, 0.0], // Z=34
    [2.0, 5.0, 0.0], // Z=35
    [2.0, 6.0, 0.0], // Z=36
    [1.0, 0.0, 0.0], // Z=37
    [2.0, 0.0, 0.0], // Z=38
    [2.0, 0.0, 1.0], // Z=39
    [2.0, 0.0, 2.0], // Z=40
    [2.0, 0.0, 3.0], // Z=41
    [2.0, 0.0, 4.0], // Z=42
    [2.0, 0.0, 5.0], // Z=43
    [2.0, 0.0, 6.0], // Z=44
    [2.0, 0.0, 7.0], // Z=45
    [2.0, 0.0, 8.0], // Z=46
    [2.0, 0.0, 9.0], // Z=47
    [2.0, 0.0, 0.0], // Z=48
    [2.0, 1.0, 0.0], // Z=49
    [2.0, 2.0, 0.0], // Z=50
    [2.0, 3.0, 0.0], // Z=51
    [2.0, 4.0, 0.0], // Z=52
    [2.0, 5.0, 0.0], // Z=53
    [2.0, 6.0, 0.0], // Z=54
    [1.0, 0.0, 0.0], // Z=55
    [2.0, 0.0, 0.0], // Z=56
    [2.0, 0.0, 1.0], // Z=57
    [2.0, 0.0, 1.0], // Z=58
    [2.0, 0.0, 1.0], // Z=59
    [2.0, 0.0, 1.0], // Z=60
    [2.0, 0.0, 1.0], // Z=61
    [2.0, 0.0, 1.0], // Z=62
    [2.0, 0.0, 1.0], // Z=63
    [2.0, 0.0, 1.0], // Z=64
    [2.0, 0.0, 1.0], // Z=65
    [2.0, 0.0, 1.0], // Z=66
    [2.0, 0.0, 1.0], // Z=67
    [2.0, 0.0, 1.0], // Z=68
    [2.0, 0.0, 1.0], // Z=69
    [2.0, 0.0, 1.0], // Z=70
    [2.0, 0.0, 1.0], // Z=71
    [2.0, 0.0, 2.0], // Z=72
    [2.0, 0.0, 3.0], // Z=73
    [2.0, 0.0, 4.0], // Z=74
    [2.0, 0.0, 5.0], // Z=75
    [2.0, 0.0, 6.0], // Z=76
    [2.0, 0.0, 7.0], // Z=77
    [2.0, 0.0, 8.0], // Z=78
    [2.0, 0.0, 9.0], // Z=79
    [2.0, 0.0, 0.0], // Z=80
    [2.0, 1.0, 0.0], // Z=81
    [2.0, 2.0, 0.0], // Z=82
    [2.0, 3.0, 0.0], // Z=83
    [2.0, 4.0, 0.0], // Z=84
    [2.0, 5.0, 0.0], // Z=85
    [2.0, 6.0, 0.0], // Z=86
];
