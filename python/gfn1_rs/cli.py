# SPDX-License-Identifier: GPL-3.0-or-later
"""Installed entry points execute the same Rust drivers as the Cargo binaries."""
import sys
from ._native import cli_main


def main():
    return cli_main(sys.argv[1:], False)


def train_main():
    return cli_main(sys.argv[1:], True)
