# SPDX-License-Identifier: GPL-3.0-or-later
"""In-process native Rust training, shared by Python and ASE workflows."""
import json
from pathlib import Path
from ._native import training_main


def create_ml_model(output, species, centers_bohr, cutoff_bohr=6.0,
                    width_bohr=1.0, maximum_level_shift_hartree=0.05):
    """Write zero-initialized native radial heads; no process is spawned."""
    training_main(["new", "--species", ",".join(map(str, species)),
                   "--centers", ",".join(map(str, centers_bohr)),
                   "--cutoff", str(cutoff_bohr), "--width", str(width_bohr),
                   "--limit", str(maximum_level_shift_hartree), "--output", str(output)])
    return Path(output)


def train_ml_model(initial_model, data, output, report, *, epochs=100,
                   learning_rate=1e-3, **options):
    """Fit through the native Rust engine and return its full validation report.

    Additional keyword names match CLI flags with underscores: e.g.
    ``gradient_weight=0``, ``train_correction=True``, ``log_file='fit.jsonl'``.
    File/data schemas, all target units and defaults are shared with the CLI.
    """
    args = ["fit", "--model", str(initial_model), "--data", str(data),
            "--output", str(output), "--report", str(report),
            "--epochs", str(epochs), "--learning-rate", str(learning_rate)]
    for key, value in options.items():
        if value is None or value is False:
            continue
        args.append("--" + key.replace("_", "-"))
        if value is not True:
            args.append(str(value))
    training_main(args)
    return json.loads(Path(report).read_text(encoding="utf-8"))
