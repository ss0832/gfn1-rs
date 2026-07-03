# SPDX-License-Identifier: GPL-3.0-or-later
"""ASE interface for the native Rust GFN1-RS implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from .native import Gfn1NativeCalculator, default_param_path


class GFN1RSCalculator(Calculator):
    """ASE calculator backed by the Rust-native GFN1-RS engine.

    ASE units are used at the Python boundary: Angstrom, eV, and
    eV/Angstrom. Periodic cells are supported: a single point with
    ``any(atoms.pbc)`` runs the Gamma / k-point PBC path (energy, forces, and
    stress), and :meth:`optimize_native` relaxes the atomic positions at fixed
    cell. Because the periodic **stress** tensor is provided, ASE's variable-cell
    barostats work too — e.g. constant-pressure MD via ``ase.md.npt.NPT`` and
    variable-cell relaxation via ``ase.constraints.ExpCellFilter`` (ASE drives the
    cell; this calculator supplies energy/forces/stress each step). Only the
    *native* L-BFGS (``optimize_native``) is fixed-cell.
    """

    # `energy` IS the finite-temperature (Mermin) free energy E - T*S_elec — the force-consistent
    # quantity the forces/stress are the gradient of (it reduces to the internal energy at
    # T_elec = 0). We deliberately do NOT expose a separate ASE `free_energy`: that name collides
    # with the *vibrational/thermodynamic* free energy from a frequency analysis and is confusing.
    # `stress` (periodic) is advertised so ASE's variable-cell barostats — e.g. the NPT ensemble
    # (`ase.md.npt.NPT`) and `ase.constraints.ExpCellFilter`/`UnitCellFilter` — accept this calculator.
    implemented_properties = ["energy", "forces", "charges", "stress"]

    default_parameters = {
        "param_path": None,
        "charge": 0.0,
        "multiplicity": None,
        # v0.4.3: spin-polarized GFN1 ("spGFN1") — collinear spin-DFTB spin term from the tblite
        # atomic spin constants W. Only affects OPEN-shell systems (set `multiplicity`); a closed-
        # shell singlet is byte-identical to plain GFN1. Non-periodic energy + analytic forces.
        "spin_polarization": False,
        # v0.4.3: DFT+U / +U+V on the correlated (transition-metal d) shell — the
        # orbital-resolved self-interaction penalty GFN1 lacks. Requires
        # spin_polarization=True (open-shell path). Set hubbard_u_linear_response=True
        # for NON-EMPIRICAL U (and V) computed by linear response — no fitted params.
        "plus_u": False,
        "hubbard_u": [],                  # [(Z, U_Hartree), ...] fixed-U mode
        "plus_u_v": False,
        "hubbard_v": [],                  # [(Za, Zb, V_Hartree), ...] fixed-V mode
        "hubbard_v_cutoff": 10.0,         # bohr
        "hubbard_u_linear_response": False,
        "plus_u_all_d": False,            # +U on all d shells (incl. main-group d), not just TM

        "max_scc": 250,
        "energy_tolerance": 1.0e-6,
        "charge_tolerance": 2.0e-5,
        "mixing": 0.4,
        "scc_broyden": True,
        "scc_broyden_size": 250,
        "electronic_temperature": 300.0,
        "nprim": 0,
        "eigen_tolerance": 1.0e-12,
        "enable_dispersion": True,
        "d3_reference_path": None,
        # v0.4.1: experimental non-periodic self-consistent D4 dispersion.
        # When True, D4 replaces D3 whenever enable_dispersion is True. a1/a2/s8
        # are read from param_gfn1-xtb.txt; d4_s9 defaults to the GFN2 value 5.0.
        "experimental_d4": False,
        "d4_cutoff": None,
        "d4_cn_cutoff": None,
        "d4_atm": True,
        "d4_atm_cutoff": None,
        "d4_s9": None,
        "enable_cn_hamiltonian": True,
        # v0.1.2: external field + SCC convergence controls.
        "electric_field": None,        # (Ex, Ey, Ez) in atomic units, or None
        "level_shift": 0.0,            # virtual level shift (Hartree)
        "scc_accelerator": None,       # "broyden" | "linear" | "cdiis" | "newton"
        # v0.1.7: experimental parameter-free mDFTB2 multipole correction. Energy + analytic forces,
        # both non-periodic AND periodic (v0.2.2): for a cell, the multipole runs through the
        # k-point SCC at arbitrary rank (`multipole_order`; default dipole+quadrupole) — the moments
        # are mixed jointly with the charges (QCore generalized-Ewald field). Periodic energy,
        # analytic forces, AND analytic stress are all FD-validated, so an `ExpCellFilter`
        # variable-cell run with `multipole=True` is supported. Off by default -> identical to GFN1.
        "multipole": False,
        # experimental octupole extension of the mDFTB2 multipole electrostatics
        # (requires multipole; only nonzero for atoms with d functions). Off by default.
        "multipole_octupole": False,
        # experimental first-order field-dipole coupling (Stage 3): with an external
        # electric field on, couples the mDFTB2 atomic dipoles to the field
        # (E_field += -E . sum_A d_A) and folds sum_A d_A into the reported dipole.
        # Requires multipole + an electric field. Off by default.
        "field_multipole": False,
        # experimental third-order on-site multipole electrostatics: adds the
        # charge*dipole^2 and charge*quad^2 on-site cross terms (parameter-free).
        # Requires multipole. Off by default.
        "multipole_third_order": False,
        # experimental richer secondary-basis on-site moments: a built-in name
        # ("cc-pVDZ"/"cc-pVTZ"/"cc-pVQZ"/"cc-pV5Z") or a secondary-basis file path.
        # When set (requires multipole), the dipole/quad moment integrals are evaluated
        # over the node-correct secondary basis. None (default) = primary-basis moments.
        "multipole_secondary_basis": None,
        # v0.2.0: HIGHEST ATOMIC MULTIPOLE RANK (the angular electrostatics). How to pick:
        #   rank 1-2 (dipole+quad)      -> multipole=True,  multipole_order=0  (default)
        #   rank 1-3 (+octupole)        -> + multipole_octupole=True (still multipole_order=0)
        #   rank 1-n, n>=4 (+16-pole..) -> multipole_order=n
        # Ranks <=3 use the byte-compatible legacy paths (the booleans pick them; order<4 is
        # ignored there). order>=4 enables the unified parameter-free arbitrary-rank path
        # (mixes atomic moments of ranks 1..n self-consistently). Requires multipole; experimental;
        # cost grows with rank. Independent of charge_order (the radial monopole expansion).
        "multipole_order": 0,
        # v0.2.1: experimental PER-RANK multipole x charge cross terms. A list whose entry l-1 is
        # the max on-site CHARGE order coupled to the rank-l (2^l-pole) atomic multipole, via the
        # breathing-radius Taylor expansion of 1/2 g_l(eta(q))(m_l.m_l). [] (default) = no cross
        # terms (just the base 2nd-order multipole). E.g. [6, 4, 2, 2] = dipole->6th, quadrupole->4th,
        # octupole/hexadecapole->off. Each entry must satisfy order <= 2l+3 (the rank-l self-energy
        # terminates there; a higher value is a hard error, never silently truncated). Requires
        # multipole and multipole_order >= the highest rank carrying a cross term (forces the generic
        # arbitrary-rank path on). Parameter-free; self-consistent energy + analytic gradient.
        "multipole_charge_order": [],
        # v0.2.0: experimental parameter-free long-range Fock exchange (MFX, LC-DFTB style).
        # When True, the Mulliken-approximated long-range exact-exchange kernel K[dP] (from the
        # hardness-derived gamma^lr + HardnessPairwise omega) is added self-consistently to the SCC
        # Fock on dP = P - P0 (neutral-atom reference) and its energy 1/2 Tr[dP K[dP]] to the total.
        # Off by default (= stock GFN1); non-periodic. Self-consistency uses commutator (Pulay) DIIS
        # on the exchange density and auto-caps the charge `mixing` so it converges out-of-the-box;
        # the analytic forces are implemented.
        "lr_exchange": False,
        # v0.2.0: experimental on-site Fock-exchange (OFX) correction layered on MFX. When True
        # (implies lr_exchange), the same-atom exchange is upgraded from the Mulliken approximation
        # to the *exact* one-center long-range two-electron integrals via the difference kernel
        # K_OFX = K_onsite,refined^lr - K_onsite,Mulliken^lr (no double count). The real STO-nG
        # one-center ERIs are built once per element (geometry-independent; cached) and contracted
        # with dP each SCC iteration. Adds no explicit force (one-center => translation-invariant);
        # forces flow through the OFX-relaxed density. Off by default; non-periodic.
        "onsite_exchange": False,
        # v0.2.0: experimental dynamic (geometry-adaptive) range separation for the long-range Fock
        # exchange (the LocalGeometry omega scheme). When True (implies lr_exchange), each atom's
        # screening is omega_A = eta_A / s_A with the parameter-free size factor s_A = (1+CN_A)^(-1/3)
        # from the GFN1 coordination number -- a more-coordinated atom screens at shorter range. False
        # (default) keeps the geometry-independent HardnessPairwise omega (= this at CN=0). The
        # analytic gradient includes the d(omega)/dR reorganization force. Non-periodic.
        "dynamic_omega": False,
        # v0.2.0: use the experimental Trust-Region Augmented Hessian (TRAH) second-order SCF for the
        # exchange-augmented SCC instead of commutator DIIS. TRAH minimizes the energy directly over
        # orbital rotations with a matrix-free Newton/trust-region step (robust where DIIS on the
        # off-diagonal exchange Fock stalls). Only affects the exchange path (lr_exchange on, multipole
        # off); closed-shell/gapped, integer occupations; non-periodic. Off by default (= DIIS driver).
        "scf_trah": False,
        # HIGHEST ORDER of the isotropic on-site charge (monopole dq) expansion -- the RADIAL
        # counterpart of multipole_order (the angular multipoles), independent of it (no multipole
        # flag needed). 3 (default) = stock GFN1 (2nd Klopman-Ohno + 3rd DFTB3). n>=4 adds the
        # experimental parameter-free Linear Breathing-Radius terms E_k = sum_A (1/k) X_k dq_A^k for
        # 4<=k<=n, X_k = (gamma/(k-1))(2*Gamma/gamma)^(k-2) (deterministic; no fitting).
        # TIP: set charge_order=4 together with lr_exchange/onsite_exchange on small-gap or metallic
        # systems -- the convex quartic bounds the unbounded cubic, so the long-range exchange no
        # longer collapses the density (e.g. it makes dppf-PdCl2 + MFX converge to a physical state).
        "charge_order": 3,
        # v0.2.2: multipole rank-continuation ("rank ladder"). When set (an int base rank) and
        # multipole_order >= 4, the high-rank multipole SCC is converged one rank at a time from
        # this base up to multipole_order, warm-starting each rank from the previous -- robust for
        # 16-pole+ multipole SCCs that a cold direct run struggles to converge. None = direct.
        # E.g. multipole_order=5, multipole_rank_ladder_base=3 -> octupole -> 16-pole -> 32-pole.
        "multipole_rank_ladder_base": None,
        # v0.4.2: experimental CAMM-on-mDFTB2 anisotropic electrostatics. "mdftb2" (default) keeps
        # the current mDFTB2 off-site multipole; "camm_on_mdftb2" replaces the off-site term by a
        # GFN2-style CAMM/AES (q-mu, q-Theta, mu-mu) on cumulative atomic multipole moments while
        # keeping the mDFTB on-site penalty (no double counting). Requires multipole=True.
        "multipole_model": "mdftb2",
        # CAMM range factor kappa (camm_on_mdftb2 only; PRIMARY lever). Scales the erf-cloud width
        # sigma_AB = kappa * sigma^HP, tuning the short-range damping range-selectively while leaving
        # the long-range 1/R^n multipole tail unchanged. Default 1.0 (parameter-free hardness width).
        "camm_damp": 1.0,
        # CAMM AES amplitude s_AES (camm_on_mdftb2 only; secondary/diagnostic lever). Scales the whole
        # AES uniformly -- note it cannot fix short-range over-attraction without weakening the correct
        # long-range tail (unlike camm_damp). Default 1.0.
        "camm_aes_scale": 1.0,
        # CAMM on-site penalty scale s_onsite (camm_on_mdftb2 only). Scales the on-site mDFTB
        # self-energy penalty fed the cumulative moments -- a lever DISTINCT from kappa (which only
        # damps the off-site AES). s_onsite < 1 tempers the cumulative-moment over-penalization that
        # dominates e.g. the halogen-bond over-binding. Default 1.0 (byte-identical to un-scaled).
        "camm_onsite_scale": 1.0,
        # Named CAMM-on-mDFTB2 preset ("polar" | "halogen" | "halogen-v1" | "halogen-allgrad" |
        # "sigma-hole"); fills per-element κ + s_onsite from the optimized table (the only way to set
        # element-specific κ/s_onsite from Python). "sigma-hole" (v0.4.4) also sets per-element
        # s_onsite. Implies multipole + camm_on_mdftb2. None disables. See docs/sigma_hole_preset.md.
        "camm_preset": None,
        # v0.1.3: periodic Monkhorst-Pack mesh ((a,b,c); None/[1,1,1] -> Gamma).
        "kgrid": None,
        # v0.1.5: path to a GFN1-xTB-cc-pVDZ secondary-basis file. When set, the
        # magnetic methods use the node-correct GFN1-xTB-M1 kinetic-energy
        # correction; None selects single-basis M0.
        "m1_basis_path": None,
    }

    def __init__(
        self,
        restart: str | None = None,
        label: str | None = None,
        atoms=None,
        **kwargs: Any,
    ) -> None:
        Calculator.__init__(self, restart=restart, label=label, atoms=atoms, **kwargs)
        self._native = self._make_native_calculator()

    def set(self, **kwargs: Any) -> dict[str, Any]:
        changed = Calculator.set(self, **kwargs)
        if changed:
            self._native = self._make_native_calculator()
        return changed

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes) -> None:
        Calculator.calculate(self, atoms, properties, system_changes)
        assert self.atoms is not None

        requested = set(properties)
        need_forces = "forces" in requested
        need_stress = "stress" in requested
        numbers = self.atoms.get_atomic_numbers().astype(np.uint8).tolist()
        positions = np.asarray(self.atoms.get_positions(), dtype=float).tolist()

        if any(self.atoms.get_pbc()):
            from ._native import BOHR_TO_ANGSTROM, HARTREE_TO_EV

            kgrid = self.parameters.kgrid
            result = self._native.calculate_periodic(
                numbers=numbers,
                positions=positions,
                cell=np.asarray(self.atoms.get_cell(), dtype=float).tolist(),
                pbc=tuple(bool(p) for p in self.atoms.get_pbc()),
                kgrid=None if kgrid is None else tuple(int(k) for k in kgrid),
                unit="angstrom",
                compute_gradient=need_forces or need_stress,
                compute_stress=need_stress,
            )
            if need_stress and result.stress is not None:
                # Native stress is 3x3 in Hartree/bohr^3; convert to eV/Angstrom^3
                # and pack into the ASE Voigt 6-vector (xx, yy, zz, yz, xz, xy).
                factor = HARTREE_TO_EV / (BOHR_TO_ANGSTROM**3)
                s = np.asarray(result.stress, dtype=float) * factor
                self.results["stress"] = np.array(
                    [s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]]
                )
        else:
            result = self._native.calculate(
                numbers=numbers,
                positions=positions,
                unit="angstrom",
                compute_gradient=need_forces,
            )
        # `energy` is the finite-temperature (Mermin) free energy E - T*S_elec (= the internal
        # energy at T_elec = 0): the quantity the forces/stress differentiate. The plain internal
        # energy is still available via `results["native_energy_terms_ev"]["total_internal"]`. No ASE
        # `free_energy` key (it would be ambiguous with the vibrational free energy).
        self.results["energy"] = float(result.energy_ev)
        self.results["charges"] = np.asarray(result.charges, dtype=float)
        self.results["native_dipole_au"] = np.asarray(result.dipole, dtype=float)
        self.results["native_energy_terms_hartree"] = dict(result.energy_terms_hartree())
        self.results["native_energy_terms_ev"] = dict(result.energy_terms_ev())
        self.results["native_converged"] = bool(result.converged)
        self.results["native_iterations"] = int(result.iterations)
        if result.forces_ev_per_angstrom is not None:
            self.results["forces"] = np.asarray(result.forces_ev_per_angstrom, dtype=float)

    def optimize_native(
        self,
        atoms=None,
        *,
        max_iterations: int = 250,
        gradient_tolerance: float = 1.0e-4,
        step_tolerance: float = 1.0e-7,
        history: int = 12,
        max_atom_step: float = 0.30,
    ):
        """Run the Rust-native L-BFGS optimizer and update the ASE atoms.

        For a periodic cell (``any(atoms.get_pbc())``) this is a **fixed-cell
        Gamma-point** optimization: the atomic positions relax while the lattice is
        held fixed (the gradient routes through the PBC path). Variable-cell
        relaxation is not yet supported.
        """

        atoms = self.atoms if atoms is None else atoms
        if atoms is None:
            raise RuntimeError("no ASE Atoms object supplied or attached")
        kwargs = dict(
            numbers=atoms.get_atomic_numbers().astype(np.uint8).tolist(),
            positions=np.asarray(atoms.get_positions(), dtype=float).tolist(),
            unit="angstrom",
            max_iterations=int(max_iterations),
            gradient_tolerance=float(gradient_tolerance),
            step_tolerance=float(step_tolerance),
            history=int(history),
            max_atom_step=float(max_atom_step),
        )
        if any(atoms.get_pbc()):
            # Fixed-cell periodic (Gamma-point) optimization: hand the cell to the
            # native optimizer so the L-BFGS gradient uses the periodic path.
            kwargs["cell"] = np.asarray(atoms.get_cell(), dtype=float).tolist()
            kwargs["pbc"] = tuple(bool(p) for p in atoms.get_pbc())
        result = self._native.optimize(**kwargs)
        atoms.set_positions(np.asarray(result.positions_angstrom, dtype=float))
        self.reset()
        return result

    def _make_native_calculator(self) -> Gfn1NativeCalculator:
        p = self.parameters
        param_path = p.param_path or default_param_path()
        return Gfn1NativeCalculator(
            param_path=str(Path(param_path)),
            charge=float(p.charge),
            multiplicity=None if p.multiplicity is None else int(p.multiplicity),
            spin_polarization=bool(p.spin_polarization),
            plus_u=bool(p.plus_u),
            hubbard_u=[(int(z), float(u)) for z, u in p.hubbard_u],
            plus_u_v=bool(p.plus_u_v),
            hubbard_v=[(int(a), int(b), float(v)) for a, b, v in p.hubbard_v],
            hubbard_v_cutoff=float(p.hubbard_v_cutoff),
            hubbard_u_linear_response=bool(p.hubbard_u_linear_response),
            plus_u_all_d=bool(p.plus_u_all_d),
            max_scc=int(p.max_scc),
            energy_tolerance=float(p.energy_tolerance),
            charge_tolerance=float(p.charge_tolerance),
            mixing=float(p.mixing),
            scc_broyden=bool(p.scc_broyden),
            scc_broyden_size=int(p.scc_broyden_size),
            electronic_temperature=float(p.electronic_temperature),
            nprim=int(p.nprim),
            eigen_tolerance=float(p.eigen_tolerance),
            enable_dispersion=bool(p.enable_dispersion),
            d3_reference_path=None
            if p.d3_reference_path is None
            else str(Path(p.d3_reference_path)),
            experimental_d4=bool(p.experimental_d4),
            d4_cutoff=None if p.d4_cutoff is None else float(p.d4_cutoff),
            d4_cn_cutoff=None if p.d4_cn_cutoff is None else float(p.d4_cn_cutoff),
            d4_atm=bool(p.d4_atm),
            d4_atm_cutoff=None if p.d4_atm_cutoff is None else float(p.d4_atm_cutoff),
            d4_s9=None if p.d4_s9 is None else float(p.d4_s9),
            enable_cn_hamiltonian=bool(p.enable_cn_hamiltonian),
            electric_field=None
            if p.electric_field is None
            else tuple(float(v) for v in p.electric_field),
            level_shift=float(p.level_shift),
            scc_accelerator=p.scc_accelerator,
            multipole=bool(p.multipole),
            multipole_octupole=bool(p.multipole_octupole),
            field_multipole=bool(p.field_multipole),
            multipole_third_order=bool(p.multipole_third_order),
            multipole_secondary_basis=p.multipole_secondary_basis,
            multipole_order=int(p.multipole_order),
            multipole_charge_order=[int(v) for v in (p.multipole_charge_order or [])],
            lr_exchange=bool(p.lr_exchange),
            onsite_exchange=bool(p.onsite_exchange),
            dynamic_omega=bool(p.dynamic_omega),
            scf_trah=bool(p.scf_trah),
            charge_order=int(p.charge_order),
            multipole_rank_ladder_base=(
                None
                if p.multipole_rank_ladder_base is None
                else int(p.multipole_rank_ladder_base)
            ),
            multipole_model=p.multipole_model,
            camm_damp=float(p.camm_damp),
            camm_aes_scale=float(p.camm_aes_scale),
            camm_onsite_scale=float(p.camm_onsite_scale),
            camm_preset=p.camm_preset,
        )

    def _numbers_positions(self, atoms):
        atoms = self.atoms if atoms is None else atoms
        if atoms is None:
            raise RuntimeError("no ASE Atoms object supplied or attached")
        if any(atoms.get_pbc()):
            raise ValueError("GFN1 spectroscopy currently supports non-PBC only")
        numbers = atoms.get_atomic_numbers().astype(np.uint8).tolist()
        positions = np.asarray(atoms.get_positions(), dtype=float).tolist()
        return numbers, positions

    def get_dipole_au(self, atoms=None):
        """Mulliken dipole moment (atomic units, e*a0) as a length-3 array."""
        numbers, positions = self._numbers_positions(atoms)
        result = self._native.calculate(numbers=numbers, positions=positions, unit="angstrom")
        return np.asarray(result.dipole, dtype=float)

    def get_polarizability(self, atoms=None):
        """Analytic static dipole polarizability (atomic units). Returns a dict
        with ``tensor`` (3x3), ``isotropic``, and ``anisotropy``."""
        numbers, positions = self._numbers_positions(atoms)
        out = self._native.polarizability(numbers=numbers, positions=positions, unit="angstrom")
        out["tensor"] = np.asarray(out["tensor"], dtype=float)
        return out

    def get_dipole_derivatives(self, atoms=None, origin=None):
        """Analytic Cartesian dipole derivatives dmu/dR (the raw IR tensor)."""
        numbers, positions = self._numbers_positions(atoms)
        out = self._native.dipole_derivatives(
            numbers=numbers, positions=positions, unit="angstrom", origin=origin
        )
        out["dipole"] = np.asarray(out["dipole"], dtype=float)
        out["ddipole_dr"] = np.asarray(out["ddipole_dr"], dtype=float)
        return out

    def get_ir_spectrum(self, atoms=None, origin=None):
        """Harmonic IR spectrum (wavenumbers + analytic intensities)."""
        numbers, positions = self._numbers_positions(atoms)
        return self._native.ir_spectrum(
            numbers=numbers, positions=positions, unit="angstrom", origin=origin
        )

    def get_hessian(self, atoms=None):
        """Analytic nuclear Hessian ``d^2E/dR_a dR_b`` as a ``(3N, 3N)`` numpy array
        (Hartree / bohr**2), atom-major Cartesian ordering. Fully analytic (no finite
        differences) -- the same Hessian used internally by the IR/Raman spectra.

        For a **periodic** cell (``any(atoms.get_pbc())``) this returns the fixed-cell
        Gamma-point (or k-point, when ``kgrid`` is set) periodic Hessian; otherwise the
        non-periodic molecular Hessian.
        """
        atoms = atoms if atoms is not None else self.atoms
        if atoms is None:
            raise RuntimeError("no ASE Atoms object supplied or attached")
        # Extract directly (do NOT use `_numbers_positions`, which rejects PBC for the
        # non-periodic spectroscopy methods; the Hessian is available for periodic cells too).
        numbers = atoms.get_atomic_numbers().astype(np.uint8).tolist()
        positions = atoms.get_positions().tolist()
        if any(atoms.get_pbc()):
            kgrid = self.parameters.kgrid
            h = self._native.hessian_periodic(
                numbers=numbers,
                positions=positions,
                cell=np.asarray(atoms.get_cell(), dtype=float).tolist(),
                pbc=tuple(bool(p) for p in atoms.get_pbc()),
                kgrid=None if kgrid is None else tuple(int(k) for k in kgrid),
                unit="angstrom",
            )
        else:
            h = self._native.hessian(numbers=numbers, positions=positions, unit="angstrom")
        return np.asarray(h, dtype=float)

    def get_vibrational_frequencies(self, atoms=None):
        """Harmonic vibrational analysis from the analytic Hessian. Returns a dict with
        ``wavenumbers`` (cm^-1; imaginary modes as negative) and ``modes`` (normal-mode
        displacements). Non-periodic.
        """
        numbers, positions = self._numbers_positions(atoms)
        return self._native.vibrational_frequencies(
            numbers=numbers, positions=positions, unit="angstrom"
        )

    def get_third_derivative_along(self, direction, atoms=None, step=1.0e-3):
        """Semi-numerical nuclear third derivative (cubic force constants) along ``direction``.

        ``direction`` is a flat ``3N`` vector (e.g. a normal mode). Returns the ``3N x 3N``
        matrix ``K[a][b] = sum_c v_c d^3E/dR_a dR_b dR_c`` (the directional derivative of the
        analytic Hessian), computed from just two analytic-Hessian evaluations. Non-periodic.
        """
        numbers, positions = self._numbers_positions(atoms)
        return self._native.third_derivative_along(
            numbers=numbers,
            positions=positions,
            direction=[float(x) for x in direction],
            unit="angstrom",
            step=float(step),
        )

    def get_third_derivative(self, atoms=None):
        """Strict **closed-form** nuclear third derivative (cubic force constants)
        ``T_abc = d^3E/dR_a dR_b dR_c``. Returns a numpy array of shape ``(3N, 3N, 3N)``
        indexed ``T[c, a, b]`` (Hartree / bohr**3), fully analytic -- no finite differences.
        Non-periodic. For best accuracy construct the calculator with a tight SCF
        (small ``energy_tolerance`` / ``charge_tolerance``). The cheaper directional
        :meth:`get_third_derivative_along` reuses just two analytic-Hessian evaluations.
        """
        numbers, positions = self._numbers_positions(atoms)
        slabs = self._native.third_derivative(
            numbers=numbers, positions=positions, unit="angstrom"
        )
        return np.asarray(slabs, dtype=float)

    def get_third_derivative_vector(self, direction, atoms=None):
        """Closed-form **Vector mode** of the cubic force constants: the directional third derivative
        ``K[a][b] = sum_c v_c T_abc`` (the derivative of the Hessian along ``direction``, e.g. a normal
        mode) as a ``(3N, 3N)`` numpy array. Returns only the ``3N x 3N`` contraction (not the full
        ``(3N, 3N, 3N)`` tensor) -- the closed-form route to use when you need a directional cubic
        constant. ``direction`` is a flat ``3N`` vector. Fully analytic. Non-periodic.
        """
        numbers, positions = self._numbers_positions(atoms)
        k = self._native.third_derivative_vector(
            numbers=numbers,
            positions=positions,
            direction=[float(x) for x in direction],
            unit="angstrom",
        )
        return np.asarray(k, dtype=float)

    def get_third_derivative_block(self, atoms_subset, atoms=None):
        """Closed-form **Block mode** of the cubic force constants over the Cartesian DOFs of the atoms
        in ``atoms_subset`` (a list of atom indices). Returns ``(dofs, T_block)`` where ``dofs`` are the
        global DOF indices (``3*atom + axis``) and ``T_block`` is an ``(m, m, m)`` numpy array
        (``m = 3*len(atoms_subset)``) indexed ``T_block[ci, ai, bi] = T[dofs[ai], dofs[bi], dofs[ci]]`` --
        an ``O(|block|^3)`` tensor for local anharmonicity over a chosen subregion. Non-periodic.
        """
        numbers, positions = self._numbers_positions(atoms)
        dofs, slabs = self._native.third_derivative_block(
            numbers=numbers,
            positions=positions,
            atoms=[int(a) for a in atoms_subset],
            unit="angstrom",
        )
        return list(dofs), np.asarray(slabs, dtype=float)

    def get_raman_spectrum(self, atoms=None, origin=None, field_step=1.0e-3):
        """Harmonic Raman spectrum (wavenumbers + activities)."""
        numbers, positions = self._numbers_positions(atoms)
        return self._native.raman_spectrum(
            numbers=numbers,
            positions=positions,
            unit="angstrom",
            origin=origin,
            field_step=float(field_step),
        )

    def get_parameter_derivatives(self, targets, atoms=None, step=1.0e-4):
        """Finite-difference parameter derivatives (dE/dp) over the given
        ``glob:`` / ``elem:`` / ``pair:`` targets."""
        numbers, positions = self._numbers_positions(atoms)
        return self._native.parameter_derivatives(
            numbers=numbers,
            positions=positions,
            targets=list(targets),
            unit="angstrom",
            step=float(step),
        )

    def get_tda(self, atoms=None, n_states=5, spin="singlet"):
        """TD-GFN1 (TDA) excited states (non-periodic). Returns a dict with
        ``excitation_energies_ev`` / ``_hartree``, ``oscillator_strengths`` and
        ``transition_dipoles``."""
        numbers, positions = self._numbers_positions(atoms)
        return self._native.tda(
            numbers=numbers,
            positions=positions,
            unit="angstrom",
            n_states=int(n_states),
            spin=str(spin),
        )

    def get_tda_gradient(
        self, atoms=None, state=0, n_states=5, spin="singlet", step=1.0e-3,
        method="semi_numerical",
    ):
        """TD-GFN1 excited-state gradient (Hartree/bohr) of the given state.

        ``method`` selects the algorithm: ``"semi_numerical"`` (default) =
        analytic ground gradient + finite difference of the frozen-amplitude
        excitation energy (recommended; exact for a tracked state, non-periodic);
        ``"fd"`` = full finite difference with root tracking (robust across state
        crossings; the only option for periodic Gamma-point cells); ``"analytic"``
        = experimental Z-vector/Lagrangian gradient (~7e-3). Periodic Atoms
        automatically fall back to ``"fd"``."""
        atoms = self.atoms if atoms is None else atoms
        if atoms is None:
            raise RuntimeError("no ASE Atoms object supplied or attached")
        numbers = atoms.get_atomic_numbers().astype(np.uint8).tolist()
        positions = np.asarray(atoms.get_positions(), dtype=float).tolist()
        kwargs = dict(
            numbers=numbers,
            positions=positions,
            unit="angstrom",
            state=int(state),
            n_states=int(n_states),
            spin=str(spin),
            step=float(step),
            method=str(method),
        )
        if any(atoms.get_pbc()):
            kwargs["cell"] = np.asarray(atoms.get_cell(), dtype=float).tolist()
            kwargs["pbc"] = tuple(bool(p) for p in atoms.get_pbc())
        return self._native.tda_gradient(**kwargs)

    def get_rotatory_strengths(self, atoms=None, n_states=5, spin="singlet", origin=None):
        """Electronic-CD rotatory strengths ``R_n = Im(mu_0n . m_n0)`` (atomic units)
        of the TD-GFN1 (TDA) excited states, about ``origin`` (default the coordinate
        origin). Returns a dict with ``excitation_energies_ev`` / ``_hartree`` and
        ``rotatory_strengths`` (all zero for an achiral molecule). Non-periodic."""
        numbers, positions = self._numbers_positions(atoms)
        return self._native.rotatory_strengths(
            numbers=numbers,
            positions=positions,
            unit="angstrom",
            n_states=int(n_states),
            spin=str(spin),
            origin=None if origin is None else tuple(float(v) for v in origin),
        )

    def get_optical_rotation(
        self, atoms=None, frequencies_ev=(0.0,), n_states=10, spin="singlet", origin=None
    ):
        """Frequency-dependent electronic optical rotation (isotropic Rosenfeld beta,
        atomic units) from the TD-GFN1 (TDA) sum over states. ``frequencies_ev`` are
        photon energies in eV (0 = static; wavelength via ``E[eV] = 1239.84/lambda[nm]``).
        Returns a dict with ``frequencies_ev`` and ``beta`` (zero for achiral molecules,
        negated for the enantiomer). Non-periodic."""
        numbers, positions = self._numbers_positions(atoms)
        return self._native.optical_rotation(
            numbers=numbers,
            positions=positions,
            unit="angstrom",
            n_states=int(n_states),
            spin=str(spin),
            frequencies_ev=[float(f) for f in frequencies_ev],
            origin=None if origin is None else tuple(float(v) for v in origin),
        )

    # ------------------------------------------------------------------
    # Magnetic field (GFN1-xTB-M0 / M1) and k-point TD-GFN1.
    # ------------------------------------------------------------------
    def _m1_path(self, m1_basis_path):
        """Resolve the M1 secondary-basis path (explicit arg overrides the
        ``m1_basis_path`` calculator parameter); returns a str or None."""
        path = m1_basis_path if m1_basis_path is not None else self.parameters.m1_basis_path
        return None if path is None else str(Path(path))

    def get_magnetic_energy(self, b_field, atoms=None, m1_basis_path=None):
        """Closed-shell magnetic (GFN1-xTB-M) SCC total energy in Hartree for a
        uniform field ``b_field = (Bx, By, Bz)`` (atomic units). Uses GFN1-xTB-M1
        when an ``m1_basis_path`` is set (here or as a calculator parameter),
        otherwise the single-basis M0. Non-periodic."""
        numbers, positions = self._numbers_positions(atoms)
        return float(
            self._native.magnetic_energy(
                numbers=numbers,
                positions=positions,
                b_field=tuple(float(v) for v in b_field),
                unit="angstrom",
                m1_basis_path=self._m1_path(m1_basis_path),
            )
        )

    def get_magnetizability(self, atoms=None, step=0.02, analytic=True, m1_basis_path=None):
        """Isotropic magnetizability ``xi_iso = -1/3 Tr d^2E/dB^2`` in SI units of
        ``1e-30 J/T^2``. ``analytic=True`` (default) uses the McWeeny density-matrix
        CP-SCC response (one magnetic SCC + cheap LAO integral derivatives, no extra
        SCF); ``analytic=False`` central-differences the energy. Uses GFN1-xTB-M1 when
        an ``m1_basis_path`` is set (strongly recommended; M0 is unreliable), else M0.
        Non-periodic."""
        numbers, positions = self._numbers_positions(atoms)
        return float(
            self._native.magnetizability(
                numbers=numbers,
                positions=positions,
                unit="angstrom",
                step=float(step),
                analytic=bool(analytic),
                m1_basis_path=self._m1_path(m1_basis_path),
            )
        )

    def get_magnetizability_diagonal(self, atoms=None, step=0.02, m1_basis_path=None):
        """Diagonal magnetizability tensor ``[xi_xx, xi_yy, xi_zz]`` (``1e-30 J/T^2``)
        from the analytic CP-SCC response; ``get_magnetizability`` returns its mean.
        Set ``m1_basis_path`` (or the calculator parameter) for M1. Non-periodic."""
        numbers, positions = self._numbers_positions(atoms)
        return np.asarray(
            self._native.magnetizability_diagonal(
                numbers=numbers,
                positions=positions,
                unit="angstrom",
                step=float(step),
                m1_basis_path=self._m1_path(m1_basis_path),
            ),
            dtype=float,
        )

    def get_magnetizability_tensor(self, atoms=None, step=0.02, m1_basis_path=None):
        """Full symmetric magnetizability tensor ``xi_ab`` as a ``(3, 3)`` array
        (``1e-30 J/T^2``) from the analytic CP-SCC response. The diagonal matches
        ``get_magnetizability_diagonal``; the off-diagonals give the anisotropy.
        Set ``m1_basis_path`` (or the calculator parameter) for M1. Non-periodic."""
        numbers, positions = self._numbers_positions(atoms)
        return np.asarray(
            self._native.magnetizability_tensor(
                numbers=numbers,
                positions=positions,
                unit="angstrom",
                step=float(step),
                m1_basis_path=self._m1_path(m1_basis_path),
            ),
            dtype=float,
        )

    def get_nmr_shielding(self, nucleus, atoms=None, m1_basis_path=None):
        """NMR nuclear magnetic shielding tensor of atom ``nucleus`` (0-based) as a
        ``(3, 3)`` array in ppm; the isotropic shielding is ``np.trace(sigma) / 3``.
        Closed-shell, non-periodic, with the common gauge origin at the shielded
        nucleus. The analytic CP-SCC magnetic-field response gives the paramagnetic
        part and a ground-state expectation the diamagnetic part. Set ``m1_basis_path``
        (or the calculator parameter) for the M1 kinetic-energy basis. Note: the GFN1
        valence-only basis omits core electrons, so absolute shieldings are not
        comparable to all-electron references (use for within-method trends)."""
        numbers, positions = self._numbers_positions(atoms)
        return np.asarray(
            self._native.nmr_shielding(
                numbers=numbers,
                positions=positions,
                nucleus=int(nucleus),
                unit="angstrom",
                m1_basis_path=self._m1_path(m1_basis_path),
            ),
            dtype=float,
        )

    def get_magnetic_polarizability(
        self, atoms=None, b_field=(0.0, 0.0, 0.0), e_step=0.002, m1_basis_path=None
    ):
        """Electric dipole polarizability ``alpha_ij(B) = dmu_i/dE_j`` (atomic units) as
        a ``(3, 3)`` array, in a uniform magnetic field ``b_field`` (atomic units), from
        the combined electric+magnetic SCC. Reduces to the field-free polarizability at
        ``B = 0``. Set ``m1_basis_path`` (or the calculator parameter) for M1.
        Non-periodic."""
        numbers, positions = self._numbers_positions(atoms)
        return np.asarray(
            self._native.magnetic_polarizability(
                numbers=numbers,
                positions=positions,
                b_field=tuple(float(v) for v in b_field),
                unit="angstrom",
                e_step=float(e_step),
                m1_basis_path=self._m1_path(m1_basis_path),
            ),
            dtype=float,
        )

    def get_cotton_mouton(self, atoms=None, e_step=0.002, b_step=0.02, m1_basis_path=None):
        """Cotton-Mouton tensor ``d^2 alpha_ij / d B_k^2`` (atomic units) as a
        ``(3, 3, 3)`` array indexed ``[k, i, j]``, driving the magnetic-field-induced
        birefringence. The first derivative ``d alpha/d B`` (MCD/Faraday) is identically
        zero in the GFN1 monopole model (``dq/dB = 0``), so only this second derivative
        is a nonzero observable. Set ``m1_basis_path`` (or the calculator parameter) for
        M1. Non-periodic."""
        numbers, positions = self._numbers_positions(atoms)
        return np.asarray(
            self._native.cotton_mouton(
                numbers=numbers,
                positions=positions,
                unit="angstrom",
                e_step=float(e_step),
                b_step=float(b_step),
                m1_basis_path=self._m1_path(m1_basis_path),
            ),
            dtype=float,
        )

    def get_mcd(self, atoms=None, e_step=0.002, b_step=0.01, m1_basis_path=None):
        """Faraday/MCD tensor ``d alpha_ij / d B_k`` (atomic units) as a ``(3, 3, 3)``
        array indexed ``[k, i, j]``. NOTE: identically zero in the GFN1 monopole model
        (``dq/dB = 0``); it is the correct general raw ``d alpha/d B`` tensor and would
        be nonzero for a length-gauge (dipole-coupled) model — see ``get_lao_dipole``.
        Set ``m1_basis_path`` (or the calculator parameter) for M1. Non-periodic."""
        numbers, positions = self._numbers_positions(atoms)
        return np.asarray(
            self._native.mcd(
                numbers=numbers,
                positions=positions,
                unit="angstrom",
                e_step=float(e_step),
                b_step=float(b_step),
                m1_basis_path=self._m1_path(m1_basis_path),
            ),
            dtype=float,
        )

    def get_angular_momentum(self, atoms=None, origin=None):
        """Raw orbital angular-momentum AO integral matrices behind the CD / magnetic-
        dipole response: a ``(3, nAO, nAO)`` array ``out[a]`` with
        ``<mu|L_a|nu> = -i * out[a, mu, nu]`` (``L = (r - O) x p`` about ``origin``; the
        orbital magnetic dipole is ``m = -1/2 L``). Non-periodic."""
        numbers, positions = self._numbers_positions(atoms)
        return np.asarray(
            self._native.angular_momentum(
                numbers=numbers,
                positions=positions,
                unit="angstrom",
                origin=None if origin is None else tuple(float(v) for v in origin),
            ),
            dtype=float,
        )

    def get_lao_dipole(self, atoms=None, b_field=(0.0, 0.0, 0.0)):
        """Raw London (GIAO) electric-dipole AO integral matrices
        ``D_c(B)_{mu nu} = <om_mu|(r_c - O)|om_nu>`` in a uniform magnetic field
        ``b_field`` (atomic units) — the length-gauge dipole behind MCD/optical rotation.
        Returns ``(re, im)`` as two ``(3, nAO, nAO)`` arrays (real, Hermitian at
        ``B = 0``). Non-periodic."""
        numbers, positions = self._numbers_positions(atoms)
        d = self._native.lao_dipole(
            numbers=numbers,
            positions=positions,
            b_field=tuple(float(v) for v in b_field),
            unit="angstrom",
        )
        return np.asarray(d["re"], dtype=float), np.asarray(d["im"], dtype=float)

    def get_magnetic_forces(self, b_field, atoms=None, step=1.0e-3, analytic=True, m1_basis_path=None):
        """Nuclear gradient/forces of the magnetic (GFN1-xTB-M) energy in a uniform
        field ``b_field`` (atomic units). ``analytic=True`` (default) uses the
        Hellmann-Feynman analytic gradient (one SCC + cheap integral derivatives);
        ``analytic=False`` uses the slower 6N+1-SCC finite difference. Set
        ``m1_basis_path`` (or the calculator parameter) for GFN1-xTB-M1 forces
        (requires ``analytic=True``). Returns a dict with ``energy_hartree``,
        ``gradient`` and ``forces`` (Hartree/bohr, numpy arrays). Non-periodic."""
        numbers, positions = self._numbers_positions(atoms)
        out = self._native.magnetic_forces(
            numbers=numbers,
            positions=positions,
            b_field=tuple(float(v) for v in b_field),
            unit="angstrom",
            step=float(step),
            analytic=bool(analytic),
            m1_basis_path=self._m1_path(m1_basis_path),
        )
        out["gradient"] = np.asarray(out["gradient"], dtype=float)
        out["forces"] = np.asarray(out["forces"], dtype=float)
        return out

    def get_tda_kpoint(
        self, atoms=None, kmesh=(2, 2, 2), n_states=5, spin="singlet", gamma_centered=True
    ):
        """Periodic TD-GFN1 (TDA) excited states over a Monkhorst-Pack ``kmesh``
        (optical q=0 transitions). Requires a periodic ASE Atoms with a cell.
        Returns a dict with ``excitation_energies_ev`` / ``_hartree`` and
        ``oscillator_strengths``."""
        atoms = self.atoms if atoms is None else atoms
        if atoms is None:
            raise RuntimeError("no ASE Atoms object supplied or attached")
        if not any(atoms.get_pbc()):
            raise ValueError("get_tda_kpoint requires a periodic ASE Atoms (set cell + pbc)")
        return self._native.tda_kpoint(
            numbers=atoms.get_atomic_numbers().astype(np.uint8).tolist(),
            positions=np.asarray(atoms.get_positions(), dtype=float).tolist(),
            cell=np.asarray(atoms.get_cell(), dtype=float).tolist(),
            kmesh=tuple(int(k) for k in kmesh),
            unit="angstrom",
            n_states=int(n_states),
            spin=str(spin),
            pbc=tuple(bool(p) for p in atoms.get_pbc()),
            gamma_centered=bool(gamma_centered),
        )

    def get_tda_kpoint_gradient(
        self,
        atoms=None,
        kmesh=(2, 2, 2),
        state=0,
        n_states=5,
        spin="singlet",
        gamma_centered=True,
        method="analytic",
        step=1.0e-3,
    ):
        """Periodic TD-GFN1 (TDA) excited-state gradient over a Monkhorst-Pack
        ``kmesh``. ``method`` is ``"analytic"`` (exact direct-CPHF gradient) or
        ``"fd"`` (finite difference). Requires a periodic ASE Atoms with a cell and
        integer (gapped) occupations. Returns a dict with ``total_energy_hartree``,
        ``excitation_energy_hartree``, ``gradient``, and ``forces`` (atomic units)."""
        atoms = self.atoms if atoms is None else atoms
        if atoms is None:
            raise RuntimeError("no ASE Atoms object supplied or attached")
        if not any(atoms.get_pbc()):
            raise ValueError(
                "get_tda_kpoint_gradient requires a periodic ASE Atoms (set cell + pbc)"
            )
        return self._native.tda_kpoint_gradient(
            numbers=atoms.get_atomic_numbers().astype(np.uint8).tolist(),
            positions=np.asarray(atoms.get_positions(), dtype=float).tolist(),
            cell=np.asarray(atoms.get_cell(), dtype=float).tolist(),
            kmesh=tuple(int(k) for k in kmesh),
            state=int(state),
            unit="angstrom",
            n_states=int(n_states),
            spin=str(spin),
            pbc=tuple(bool(p) for p in atoms.get_pbc()),
            gamma_centered=bool(gamma_centered),
            method=str(method),
            step=float(step),
        )
