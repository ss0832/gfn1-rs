// SPDX-License-Identifier: GPL-3.0-or-later
fn main() {
    std::process::exit(gfn1_rs::cli::main(
        std::env::args().skip(1).collect(),
        false,
    ));
}
