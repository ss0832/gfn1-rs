# Fragment molecular orbitals and divide-and-conquer

The Rust `fragment` module implements two distinct approximations to the
restricted, isotropic, third-order GFN1-xTB model. Both use the complete-system
coordination numbers. D3 and the halogen correction are evaluated once for the
complete system. Unsupported model combinations produce an error.

## FMO2 and FMO3

`fragment::fmo_xtb` converges the monomers in their mutual electrostatic embedding,
then evaluates embedded dimers and, for `order=3`, trimers. In addition to the
many-body internal-energy expansion it includes the charge-transfer embedding
correction

```text
Delta E_C^V = (q_C - sum_{I in C} q_I)^T Gamma q_outside_C.
```

At positive electronic temperature the same expansion applies to the electronic
free energies, with a canonical electron count in each subsystem. All monomer
charges remain coupled when differentiating. The implementation factors the
embedding Jacobian once and reuses it for the nuclear and strain right-hand
sides. It includes changes of the embedding charges, the overlap, the global
coordination numbers, the Coulomb kernel and the electronic entropy.

`CovalentCut { bda, baa, image }` is an oriented hybrid-orbital-projection boundary.
The detached atom BDA loses one reference electron in its own subsystem; the
subsystem containing BAA receives a ghost at BDA with one reference electron.
An sp3 hybrid points along the specified bond. The overlap-weighted HOP projector
enters the Fock matrix but is excluded from the physical energy. Repulsion uses
the modified reference-charge scaling. Dimers/trimers containing both endpoints
heal the cut and remove its ghost and projector. Charge-transfer corrections
include disappearing ghost charges. At most one detached bond per BDA is allowed,
and a complete valence s/p shell is required. Cuts must connect different
fragments; automatic chemical fragmentation is not implemented.

The default HOP barrier is 1e6 Hartree. A separate geometry-only constraint basis
and relative-accuracy Jacobi solve keep ordinary SCC updates from being lost
when added to that large barrier. The analytic derivative includes both motion
of the hybrid direction and the overlap factors of the projector.

```python
from gfn1_rs import Gfn1NativeCalculator
calc = Gfn1NativeCalculator(electronic_temperature=300, max_scc=1000,
                           energy_tolerance=1e-13, charge_tolerance=1e-11)
numbers = [1, 9, 1, 9, 1, 9]
positions = [[0,0,0], [.92,0,0], [0,3,0], [.92,3,0], [3,0,0], [3.92,0,0]]
fragments = [[0,1], [2,3], [4,5]]
out = calc.fmo_xtb(numbers, positions, fragments, fragment_charges=[0,0,0], order=2)
print(out["total_free_hartree"], out["forces"])
# Covalent systems additionally pass cuts=[(bda, baa), ...].
# cut_images=[(nx, ny, nz), ...] fixes their periodic bond images.
```

Atom indices are zero-based. Fragments are disjoint nonempty atom lists covering
the system exactly once. Their charges must sum to the calculator's total charge.
At zero temperature every subsystem must admit an even, gapped filling; choose
positive temperature for fractional occupations. Supplying arbitrary link
hydrogens is not equivalent to the HOP boundary implemented here.

## Density-matrix divide-and-conquer

`fragment::divide_and_conquer` diagonalizes overlapping core-plus-buffer
subsystems in the shared SCC potential. A single chemical potential conserves
the **total** electron number. Its reconstructed density uses the symmetric
core partition `W_mu,nu=(w_mu+w_nu)/2`. The free energy contains the matching
partitioned entropy trace. Individual Mulliken core contributions may be
negative; imposing positivity on each would incorrectly reject even the exact
complete-buffer limit.

The finite-buffer energy is not stationary with respect to each local orbital.
The derivative therefore includes its density and entropy response and solves
the coupled global-charge/chemical-potential equations. Using only the ordinary
GFN1 Hellmann-Feynman force would omit those terms.

```python
dc = calc.divide_and_conquer(
    numbers, positions, fragments,
    buffers=[[0,1,2,3], [0,1,2,3,4,5], [2,3,4,5]],
    fragment_charges=[0,0,0],
)
print(dc["total_free_hartree"], dc["gradient"])
```

Each frozen buffer list must contain its entire core. Its topology stays fixed
while differentiating; rebuild the lists deliberately when changing the
fragmentation. Overlapping buffers extend across covalent bonds directly. HOP
cuts belong to FMO and are not applied to DC. Zero-temperature finite buffers
are accepted only when their weighted occupations conserve electrons; otherwise
the API requests positive temperature. Complete buffers recover ordinary GFN1.

## Rust API

All calculations below use 300 K and return native atomic units. Both methods
accept the same explicit electronic options, periodic controls and partition.

```rust
use gfn1_rs::{ElectronicOptions,Gfn1Parameters,PeriodicSystem};
use gfn1_rs::fragment::{FragmentPartition,FmoOptions,DivideConquerOptions,
                        fmo_xtb,divide_and_conquer};
use gfn1_rs::pbc::PbcOptions;

fn main() -> gfn1_rs::Result<()> {
    let molecule = PeriodicSystem::from_xyz_str(
        "6\nthree HF fragments, angstrom\nH 0 0 0\nF .92 0 0\nH 0 3 0\nF .92 3 0\nH 3 0 0\nF 3.92 0 0\n",
        0.0,false)?;
    let params=Gfn1Parameters::builtin()?;
    let electronic=ElectronicOptions {electronic_temperature:300.,max_scc:1000,
        energy_tolerance:1e-13,charge_tolerance:1e-11,..Default::default()};
    let partition=FragmentPartition {atoms:vec![vec![0,1],vec![2,3],vec![4,5]],
        charges:vec![0.,0.,0.],cuts:vec![]};
    let pbc=PbcOptions::default(); // Gamma
    let fmo=fmo_xtb(&molecule,&params,&electronic,&pbc,&partition,&FmoOptions::default())?;
    let dc_options=DivideConquerOptions {buffers:vec![(0..6).collect();3],..Default::default()};
    let dc=divide_and_conquer(&molecule,&params,&electronic,&pbc,&partition,&dc_options)?;
    println!("FMO {} and DC {} Hartree",fmo.total_free,dc.total_free);
    println!("FMO forces {:?} Hartree/bohr",fmo.forces);
    // For a periodic calculation, attach a lattice in bohr to the same system:
    let mut crystal=molecule.clone();
    let a=12.*gfn1_rs::constants::ANGSTROM_TO_BOHR;
    crystal.lattice=Some(gfn1_rs::lattice::Lattice::from_vectors(
        gfn1_rs::math::Vec3::new(a,0.,0.),gfn1_rs::math::Vec3::new(0.,a,0.),
        gfn1_rs::math::Vec3::new(0.,0.,a),[true;3])?);
    let periodic=fmo_xtb(&crystal,&params,&electronic,&pbc,&partition,&FmoOptions::default())?;
    println!("stress {:?} Hartree/bohr^3",periodic.stress);
    Ok(())
}
```

To define a covalent FMO boundary, populate `partition.cuts` with
`CovalentCut { bda: carbon_a, baa: carbon_b, image: [0,0,0] }` for an actual
bond connecting two different fragments. For a bond crossing the cell boundary,
`image` selects the BAA lattice image relative to BDA and stays fixed under strain.
Set `FmoOptions { order:3, ..Default::default() }` for FMO3. DC uses frozen
`buffers` rather than covalent cuts. `compute_derivatives:false` skips response
solves; its Rust gradient/force vectors are empty and stress is `None`.

## Python native: periodic cells and covalent bonds

The molecular `calc.fmo_xtb` and `calc.divide_and_conquer` examples above use
the native API. The same input and 300 K calculator extend to periodic cells:

```python
cell = [[12.,0.,0.],[0.,12.,0.],[0.,0.,12.]]  # angstrom
periodic_fmo = calc.fmo_xtb(numbers, positions, fragments, cell=cell, order=2)
periodic_dc = calc.divide_and_conquer(
    numbers, positions, fragments, buffers=[list(range(6))]*3, cell=cell)
print(periodic_fmo["forces"], periodic_fmo["stress"])
```

For an actual covalent split, this ethane example supplies a carbon-carbon HOP
boundary explicitly (coordinates are angstrom):

```python
ethane_numbers = [6,6,1,1,1,1,1,1]
ethane_positions = [[-.77,0,0],[.77,0,0],[-1.14,1.02,0],[-1.14,-.51,.883],
                    [-1.14,-.51,-.883],[1.14,-1.02,0],[1.14,.51,.883],[1.14,.51,-.883]]
ethane_fragments = [[0,2,3,4],[1,5,6,7]]
covalent_fmo = calc.fmo_xtb(ethane_numbers, ethane_positions, ethane_fragments,
                            cuts=[(0,1)], fragment_charges=[0,0])
```

For periodic covalent cuts, add `cell=...` and one `cut_images` tuple for each
cut. Do not infer the image afresh at displaced geometries or strained cells.

## ASE API

```python
from ase import Atoms
from gfn1_rs import GFN1RSCalculator
atoms = Atoms(numbers=numbers, positions=positions)
atoms.calc = GFN1RSCalculator(electronic_temperature=300., max_scc=1000,
                             energy_tolerance=1e-13, charge_tolerance=1e-11)
fmo = atoms.calc.get_fmo_xtb(fragments, atoms=atoms, fragment_charges=[0,0,0], order=2)
dc = atoms.calc.get_divide_and_conquer(
    fragments, atoms=atoms, buffers=[list(range(6))]*3, fragment_charges=[0,0,0])
print(fmo["total_free"], fmo["forces"])  # eV, eV/angstrom

crystal = atoms.copy()
crystal.set_cell([12.,12.,12.])
crystal.pbc = True
crystal.calc = GFN1RSCalculator(electronic_temperature=300., kgrid=(1,1,1), max_scc=1000)
periodic = crystal.calc.get_fmo_xtb(fragments, atoms=crystal)
print(periodic["stress"])  # 3x3, eV/angstrom^3
```

ASE forwards `cuts`, `cut_images`, `fragment_charges`, `buffers`, response
controls and memory bounds to the same Rust methods. Only Gamma sampling is
supported. Unsuffixed `total_free`, `gradient`, `forces` and `stress` use ASE units.
`total_free_hartree`, `gradient_hartree_per_bohr`, `forces_hartree_per_bohr` and
`stress_hartree_per_bohr3` preserve native values. With derivatives disabled,
the derivative fields remain `None`. These explicit property methods return
their own fragment results; they do not switch ordinary `atoms.get_forces()`
or its cache from the calculator's usual full-system Hamiltonian to FMO/DC.

## Periodic extension, units and limits

Pass `cell` (three lattice-vector rows in the same input unit as positions) to
either Python method. The periodic extension uses real Gamma image-summed AO
matrices and three-dimensional Ewald shell electrostatics. It is a coherent
reference-cell formulation. Arbitrary k-point fragment orbitals and one-/two-
dimensional Ewald boundary conditions are not implemented; use converged
supercells for sampling. The periodic formulation and stress are this project's
extension, not a claim that the molecular FMO paper already implements PBC.

Energies are Hartree, gradients and forces are Hartree/bohr, and stress is the
3x3 matrix `(1/V) dF/d epsilon_ab` in Hartree/bohr^3 at fixed fractional
coordinates. Stress is present only for a periodic cell. With
`compute_derivatives=False`, the Python derivative fields are `None`.
FMO `shell_charges` are the converged monomer embedding charges; DC returns the
reconstructed complete-system shell charges.

These implementations share geometry matrices and bounded local solves. They
still retain complete reference-cell matrices and a dense response Jacobian;
they do not claim linear memory or asymptotic linear scaling. `max_ao` defaults
to 512; FMO also bounds the subsystem count with `max_clusters`. A larger order
or buffer is a convergence choice, not a universal accuracy guarantee.

Verification is in `tests/fragment.rs` and the geometry component gate. The
complete FMO3/DC limits recover the reference energy and gradient at 0 K and
30000 K. Molecular and periodic FMO2/DC compare every force coordinate, plus all
nine periodic strain components, against two-step extrapolated energy
differences. Covalent HOP is tested separately. Difference stencils keep the
neighbor topology fixed: crossing a hard cutoff does not define a smooth
derivative.

## References

- R. Einsele and R. Mitric, *FMO-xTB: Fragment molecular orbital method with
  GFN1-xTB for large-scale quantum-mechanical simulations*,
  [arXiv:2606.28022v1](https://arxiv.org/abs/2606.28022v1), including the analytic
  embedding response and covalent HOP derivations. The paper PDF was obtained
  in the validation workspace; it is not redistributed with the package.
- Reference implementation reviewed:
  [DIALECT-rs](https://github.com/dialect-rs/DIALECT-rs), commit
  `b76da09e6df205b4439ca8ee4e4b296c3d5ccc7b`, especially
  `dialect-xtb/src/fmo/scc_hop`, `fmo/scc`, and `fmo/gradients`.
  The source archive was obtained. This implementation is independently written
  in the current codebase, without importing DIALECT as a runtime dependency.
- W. Yang, *Direct calculation of electron density in density-functional theory*,
  Physical Review Letters **66**, 1438–1441 (1991),
  [doi:10.1103/PhysRevLett.66.1438](https://doi.org/10.1103/PhysRevLett.66.1438).
- W. Yang and T.-S. Lee, *A density-matrix divide-and-conquer approach for
  electronic structure calculations of large molecules*, J. Chem. Phys.
  **103**, 5674–5678 (1995),
  [doi:10.1063/1.470549](https://doi.org/10.1063/1.470549).
- Parent GFN1-xTB parameters and citations: [parameters.md](parameters.md).
