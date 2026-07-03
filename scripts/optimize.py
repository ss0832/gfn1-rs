"""Geometry optimization of an arbitrary molecule (XYZ) with the native GFN1-RS L-BFGS
optimizer, optionally with the experimental parameter-free corrections turned on.

Works on **any** XYZ file (Z = 1-86, the GFN1 range). The model corrections are opt-in flags:
  --multipole          mDFTB2 atomic dipole+quadrupole electrostatics
  --multipole-order N  arbitrary-rank multipole (N>=4; implies --multipole)
  --lr-exchange        long-range Mulliken Fock exchange (MFX)
  --onsite-exchange    + exact one-center Fock exchange (OFX); implies --lr-exchange
  --scf-trah           use the TRAH second-order SCF for the exchange path
  --charge-order N     on-site charge expansion order (3 = stock GFN1)
All are off by default, so with no flags this is plain GFN1.

Examples
--------
  python scripts/optimize.py examples/water.xyz
  python scripts/optimize.py mol.xyz --multipole --out mol_opt.xyz --traj mol_traj.xyz
  python scripts/optimize.py mol.xyz --multipole --compare        # plain-vs-corrected
  python scripts/optimize.py complex.xyz --charge -1 --multiplicity 2 --etemp 1000

Needs the built extension (`maturin develop --release --features python`) and `GFN1_XTB_PARAM`
(or pass --param). `numpy` is required; ASE is not.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

from gfn1_rs.native import Gfn1NativeCalculator, default_param_path

# Element symbol -> atomic number, Z = 1..86 (the GFN1-xTB element range).
_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn"
).split()
SYMBOL_TO_Z = {s: i + 1 for i, s in enumerate(_SYMBOLS)}


def load_xyz(path):
    """Read a plain XYZ file -> (symbols, positions[N,3] in Angstrom)."""
    lines = Path(path).read_text().splitlines()
    nat = int(lines[0].split()[0])
    syms, pos = [], []
    for ln in lines[2 : 2 + nat]:
        p = ln.split()
        syms.append(p[0])
        pos.append([float(p[1]), float(p[2]), float(p[3])])
    return syms, np.asarray(pos, dtype=float)


def numbers_for(symbols):
    try:
        return [SYMBOL_TO_Z[s] for s in symbols]
    except KeyError as exc:
        raise SystemExit(f"unsupported element {exc!s}; GFN1 covers Z = 1-86 (H-Rn)")


def make_calculator(args, *, corrections):
    """Build a native calculator. `corrections` False => plain GFN1 (all knobs off)."""
    return Gfn1NativeCalculator(
        param_path=str(args.param),
        charge=float(args.charge),
        multiplicity=args.multiplicity,
        max_scc=int(args.max_scc),
        mixing=float(args.mixing),
        electronic_temperature=float(args.etemp),
        multipole=bool(corrections and (args.multipole or args.multipole_order >= 4)),
        multipole_order=int(args.multipole_order if corrections else 0),
        lr_exchange=bool(corrections and (args.lr_exchange or args.onsite_exchange)),
        onsite_exchange=bool(corrections and args.onsite_exchange),
        scf_trah=bool(corrections and args.scf_trah),
        charge_order=int(args.charge_order if corrections else 3),
    )


def optimize(calc, numbers, pos, args, traj=None):
    return calc.optimize(
        numbers=numbers,
        positions=pos.tolist(),
        unit="angstrom",
        max_iterations=int(args.max_iter),
        gradient_tolerance=float(args.gtol),
        max_atom_step=float(args.max_step),
        history=int(args.history),
        trajectory_path=traj,
    )


def report(tag, res):
    print(f"  [{tag}] E = {res.free_energy_ev:.6f} eV   "
          f"max|grad| = {res.max_gradient:.3e}   "
          f"steps = {res.iterations}   converged = {res.converged}")


def parse_args(argv):
    p = argparse.ArgumentParser(description="GFN1-RS geometry optimization of any XYZ.")
    p.add_argument("xyz", help="input geometry (XYZ, Angstrom)")
    p.add_argument("--param", default=os.environ.get("GFN1_XTB_PARAM", default_param_path()))
    p.add_argument("--charge", type=float, default=0.0)
    p.add_argument("--multiplicity", type=int, default=None)
    p.add_argument("--etemp", type=float, default=300.0, help="electronic temperature (K)")
    p.add_argument("--max-scc", type=int, default=250)
    p.add_argument("--mixing", type=float, default=0.4)
    # model corrections (all experimental, off by default => plain GFN1)
    p.add_argument("--multipole", action="store_true")
    p.add_argument("--multipole-order", type=int, default=0)
    p.add_argument("--lr-exchange", action="store_true")
    p.add_argument("--onsite-exchange", action="store_true")
    p.add_argument("--scf-trah", action="store_true")
    p.add_argument("--charge-order", type=int, default=3)
    # optimizer controls
    p.add_argument("--max-iter", type=int, default=400)
    p.add_argument("--gtol", type=float, default=1.0e-3, help="max|grad| convergence (eV/A)")
    p.add_argument("--max-step", type=float, default=0.15)
    p.add_argument("--history", type=int, default=14)
    # output
    p.add_argument("--out", default=None, help="write the final geometry (XYZ)")
    p.add_argument("--traj", default=None, help="stream the trajectory live to this XYZ")
    p.add_argument("--compare", action="store_true",
                   help="also optimize plain GFN1 and report the on-vs-off displacement")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    syms, pos0 = load_xyz(args.xyz)
    numbers = numbers_for(syms)
    corrections_on = (args.multipole or args.multipole_order >= 4 or args.lr_exchange
                      or args.onsite_exchange or args.charge_order != 3)
    tag = "GFN1+corr" if corrections_on else "GFN1"
    print(f"optimizing {args.xyz}: {len(numbers)} atoms ({tag})")

    res = optimize(make_calculator(args, corrections=True), numbers, pos0, args, traj=args.traj)
    report(tag, res)
    if args.out:
        Path(args.out).write_text(res.to_xyz(f"{Path(args.xyz).stem} optimized ({tag})"))
        print(f"  wrote {args.out}")

    if args.compare and corrections_on:
        ref = optimize(make_calculator(args, corrections=False), numbers, pos0, args)
        report("GFN1 (plain)", ref)
        drift = np.linalg.norm(np.asarray(res.positions_angstrom)
                               - np.asarray(ref.positions_angstrom), axis=1)
        print(f"  on-vs-off displacement: max {drift.max():.3f} A, mean {drift.mean():.3f} A")


if __name__ == "__main__":
    main()
