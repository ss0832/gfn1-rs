# SPDX-License-Identifier: GPL-3.0-or-later
"""Short k-point PBC MD runs for MgO with the gfn1_rs ASE calculator.

The run uses a two-atom primitive rocksalt MgO cell and discards the first
``--equil-steps`` from the reported statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.md.verlet import VelocityVerlet

from gfn1_rs.ase import GFN1RSCalculator


def make_mgo() -> Atoms:
    """Return a primitive two-atom rocksalt MgO cell."""

    a = 4.212  # Angstrom, experimental room-temperature MgO rocksalt cell.
    cell = [
        (0.0, 0.5 * a, 0.5 * a),
        (0.5 * a, 0.0, 0.5 * a),
        (0.5 * a, 0.5 * a, 0.0),
    ]
    return Atoms(
        "MgO",
        positions=[(0.0, 0.0, 0.0), (0.5 * a, 0.0, 0.0)],
        cell=cell,
        pbc=True,
    )


def parse_kgrid(text: str) -> tuple[int, int, int]:
    parts = [int(x) for x in text.replace("x", ",").split(",") if x.strip()]
    if len(parts) != 3 or any(x < 1 for x in parts):
        raise argparse.ArgumentTypeError("kgrid must be like 2,2,2")
    return tuple(parts)


def attach_calc(atoms: Atoms, args: argparse.Namespace) -> None:
    atoms.calc = GFN1RSCalculator(
        param_path=args.param,
        kgrid=args.kgrid,
        max_scc=args.max_scc,
        energy_tolerance=args.energy_tolerance,
        charge_tolerance=args.charge_tolerance,
        electronic_temperature=args.electronic_temperature,
        mixing=args.mixing,
        scc_accelerator=args.scc_accelerator,
    )


def initialize_velocities(atoms: Atoms, temperature_k: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_k, rng=rng)
    Stationary(atoms)
    ZeroRotation(atoms)


def voigt_to_pressure_gpa(stress: np.ndarray) -> float:
    # ASE stress convention: pressure = -trace(stress) / 3.
    return float(-(stress[0] + stress[1] + stress[2]) / (3.0 * units.GPa))


def density_g_cm3(atoms: Atoms) -> float:
    # amu / A^3 -> g / cm^3
    return float(atoms.get_masses().sum() * 1.66053906660 / atoms.get_volume())


def cell_lengths(atoms: Atoms) -> tuple[float, float, float]:
    c = atoms.cell.cellpar()
    return float(c[0]), float(c[1]), float(c[2])


def sample_observables(atoms: Atoms, step: int, elapsed_s: float) -> dict[str, float]:
    epot = float(atoms.get_potential_energy())
    ekin = float(atoms.get_kinetic_energy())
    stress = atoms.get_stress(include_ideal_gas=True)
    a, b, c = cell_lengths(atoms)
    return {
        "step": step,
        "time_fs": step * sample_observables.dt_fs,
        "elapsed_s": elapsed_s,
        "epot_eV": epot,
        "ekin_eV": ekin,
        "etot_eV": epot + ekin,
        "temperature_K": float(atoms.get_temperature()),
        "pressure_GPa": voigt_to_pressure_gpa(stress),
        "volume_A3": float(atoms.get_volume()),
        "density_g_cm3": density_g_cm3(atoms),
        "cell_a_A": a,
        "cell_b_A": b,
        "cell_c_A": c,
    }


sample_observables.dt_fs = 0.5


def berendsen_scale_velocities(atoms: Atoms, args: argparse.Namespace) -> None:
    old_temperature = max(float(atoms.get_temperature()), 1.0e-12)
    factor = math.sqrt(
        1.0
        + (args.temperature_k / old_temperature - 1.0)
        * (args.dt_fs / args.thermostat_tau_fs)
    )
    factor = min(1.1, max(0.9, factor))
    atoms.set_momenta(factor * atoms.get_momenta())


def berendsen_scale_cell(atoms: Atoms, args: argparse.Namespace) -> None:
    stress = atoms.get_stress(voigt=False, include_ideal_gas=True)
    old_pressure = float(-np.trace(stress) / 3.0)
    target_pressure = args.pressure_gpa * units.GPa
    compressibility = 1.0 / (args.bulk_modulus_gpa * units.GPa)
    scale = 1.0 - (args.dt_fs / args.barostat_tau_fs) * compressibility * (
        target_pressure - old_pressure
    ) / 3.0
    atoms.set_cell(scale * atoms.get_cell(), scale_atoms=True)


def cached_forces_from_stress(atoms: Atoms) -> np.ndarray:
    atoms.get_stress(include_ideal_gas=True)
    forces = atoms.calc.results.get("forces")
    if forces is None:
        forces = atoms.get_forces(md=True)
    return np.asarray(forces, dtype=float)


def fast_npt_step(atoms: Atoms, forces: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    # Single-evaluation Berendsen NPT step.  The pressure uses cached potential
    # stress plus current ideal-gas stress; the new geometry is evaluated once at
    # the end, yielding both next-step stress and final half-kick forces.
    dt = args.dt_fs * units.fs
    berendsen_scale_velocities(atoms, args)
    berendsen_scale_cell(atoms, args)

    p = atoms.get_momenta()
    p += 0.5 * dt * forces
    p -= p.sum(axis=0) / float(len(p))
    atoms.set_positions(atoms.get_positions() + dt * p / atoms.get_masses()[:, np.newaxis])
    atoms.set_momenta(p)

    new_forces = cached_forces_from_stress(atoms)
    atoms.set_momenta(atoms.get_momenta() + 0.5 * dt * new_forces)
    return new_forces


def write_xyz(path: Path, atoms: Atoms, label: str) -> None:
    path.write_text(
        "\n".join(
            [
                str(len(atoms)),
                (
                    f'Lattice="{atoms.cell[0,0]:.10f} {atoms.cell[0,1]:.10f} {atoms.cell[0,2]:.10f} '
                    f'{atoms.cell[1,0]:.10f} {atoms.cell[1,1]:.10f} {atoms.cell[1,2]:.10f} '
                    f'{atoms.cell[2,0]:.10f} {atoms.cell[2,1]:.10f} {atoms.cell[2,2]:.10f}" '
                    f'pbc="T T T" {label}'
                ),
                *(
                    f"{sym:2s} {pos[0]:18.10f} {pos[1]:18.10f} {pos[2]:18.10f}"
                    for sym, pos in zip(atoms.get_chemical_symbols(), atoms.positions)
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_ensemble(
    name: str,
    atoms: Atoms,
    args: argparse.Namespace,
    outdir: Path,
) -> dict[str, object]:
    initialize_velocities(atoms, args.temperature_k, args.seed + (0 if name == "nve" else 1000))
    attach_calc(atoms, args)
    sample_observables.dt_fs = args.dt_fs

    fast_npt = name == "npt" and args.npt_integrator == "fast"
    if name == "nve":
        dyn = VelocityVerlet(
            atoms,
            timestep=args.dt_fs * units.fs,
            logfile=None,
        )
    elif name == "npt" and not fast_npt:
        dyn = NPTBerendsen(
            atoms,
            timestep=args.dt_fs * units.fs,
            temperature_K=args.temperature_k,
            pressure_au=args.pressure_gpa * units.GPa,
            taut=args.thermostat_tau_fs * units.fs,
            taup=args.barostat_tau_fs * units.fs,
            compressibility_au=1.0 / (args.bulk_modulus_gpa * units.GPa),
            logfile=None,
        )
    elif fast_npt:
        dyn = None
    else:
        raise ValueError(name)

    rows: list[dict[str, float]] = []
    total_steps = args.equil_steps + args.prod_steps
    start = time.perf_counter()
    forces = cached_forces_from_stress(atoms) if fast_npt else atoms.get_forces()
    for step in range(1, total_steps + 1):
        if fast_npt:
            forces = fast_npt_step(atoms, forces, args)
        else:
            forces = dyn.step(forces)
        if step % args.sample_stride == 0:
            rows.append(sample_observables(atoms, step, time.perf_counter() - start))
        if args.progress and (step == 1 or step % args.progress == 0 or step == total_steps):
            print(f"{name.upper()} step {step}/{total_steps}", flush=True)

    csv_path = outdir / f"{name}_timeseries.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    measured = [r for r in rows if r["step"] > args.equil_steps]
    summary = summarize(name, measured, args, atoms, time.perf_counter() - start)
    summary["timeseries_csv"] = str(csv_path)
    write_xyz(outdir / f"{name}_final.xyz", atoms, f"{name.upper()} final structure")
    return summary


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def summarize(
    name: str,
    rows: list[dict[str, float]],
    args: argparse.Namespace,
    atoms: Atoms,
    elapsed_s: float,
) -> dict[str, object]:
    out: dict[str, object] = {
        "ensemble": name.upper(),
        "equil_steps_discarded": args.equil_steps,
        "measurement_steps": args.prod_steps,
        "sampled_points": len(rows),
        "dt_fs": args.dt_fs,
        "temperature_target_K": args.temperature_k,
        "kgrid": args.kgrid,
        "elapsed_s": elapsed_s,
    }
    for key in [
        "epot_eV",
        "ekin_eV",
        "etot_eV",
        "temperature_K",
        "pressure_GPa",
        "volume_A3",
        "density_g_cm3",
        "cell_a_A",
    ]:
        vals = [float(r[key]) for r in rows]
        out[f"{key}_mean"] = mean(vals)
        out[f"{key}_std"] = std(vals)
    if rows:
        out["etot_eV_drift_per_ps"] = (
            (rows[-1]["etot_eV"] - rows[0]["etot_eV"])
            / ((rows[-1]["time_fs"] - rows[0]["time_fs"]) / 1000.0)
        )
        out["volume_A3_last"] = rows[-1]["volume_A3"]
        out["pressure_GPa_last"] = rows[-1]["pressure_GPa"]
    out["final_volume_A3"] = float(atoms.get_volume())
    out["final_density_g_cm3"] = density_g_cm3(atoms)
    return out


def write_markdown(path: Path, summaries: list[dict[str, object]], args: argparse.Namespace) -> None:
    lines = [
        "# MgO k-point PBC MD summary",
        "",
        f"- System: primitive rocksalt MgO, 2 atoms, kgrid={args.kgrid}",
        f"- Run length: {args.equil_steps} equilibration + {args.prod_steps} production steps",
        f"- Time step: {args.dt_fs} fs; target temperature: {args.temperature_k} K",
        f"- NPT pressure target: {args.pressure_gpa} GPa; bulk modulus for Berendsen compressibility: {args.bulk_modulus_gpa} GPa",
        "",
        "| Ensemble | T mean K | P mean GPa | V mean A^3 | density g/cm^3 | Etot drift eV/ps |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            "| {ensemble} | {t:.2f} +/- {ts:.2f} | {p:.3f} +/- {ps:.3f} | "
            "{v:.4f} +/- {vs:.4f} | {rho:.4f} +/- {rhos:.4f} | {drift:.4e} |".format(
                ensemble=s["ensemble"],
                t=s["temperature_K_mean"],
                ts=s["temperature_K_std"],
                p=s["pressure_GPa_mean"],
                ps=s["pressure_GPa_std"],
                v=s["volume_A3_mean"],
                vs=s["volume_A3_std"],
                rho=s["density_g_cm3_mean"],
                rhos=s["density_g_cm3_std"],
                drift=s.get("etot_eV_drift_per_ps", math.nan),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", default=os.environ.get("GFN1_XTB_PARAM", "../param_gfn1-xtb.txt"))
    parser.add_argument("--outdir", default="md_mgo_kpoint_results")
    parser.add_argument("--ensemble", choices=["all", "nve", "npt"], default="all")
    parser.add_argument("--equil-steps", type=int, default=100)
    parser.add_argument("--prod-steps", type=int, default=1000)
    parser.add_argument("--dt-fs", type=float, default=0.5)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--pressure-gpa", type=float, default=0.0)
    parser.add_argument("--bulk-modulus-gpa", type=float, default=160.0)
    parser.add_argument("--thermostat-tau-fs", type=float, default=50.0)
    parser.add_argument("--barostat-tau-fs", type=float, default=500.0)
    parser.add_argument("--npt-integrator", choices=["fast", "ase"], default="fast")
    parser.add_argument("--kgrid", type=parse_kgrid, default=(2, 2, 2))
    parser.add_argument("--max-scc", type=int, default=300)
    parser.add_argument("--energy-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--charge-tolerance", type=float, default=5.0e-5)
    parser.add_argument("--electronic-temperature", type=float, default=300.0)
    parser.add_argument("--mixing", type=float, default=0.4)
    parser.add_argument("--scc-accelerator", default="broyden")
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--progress", type=int, default=100)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base = make_mgo()
    write_xyz(outdir / "mgo_initial.xyz", base, "initial primitive rocksalt MgO")
    summaries = []
    for ensemble in (["nve", "npt"] if args.ensemble == "all" else [args.ensemble]):
        summaries.append(run_ensemble(ensemble, base.copy(), args, outdir))

    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    write_markdown(outdir / "summary.md", summaries, args)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
