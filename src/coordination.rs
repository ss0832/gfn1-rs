// SPDX-License-Identifier: GPL-3.0-or-later

use crate::data_tables::covalent_radius_d3_bohr;
use crate::error::Result;
use crate::math::Vec3;
use crate::pairlist::for_each_unique_short_range_pair;
use crate::system::PeriodicSystem;

const DIST_EPS: f64 = 1.0e-12;

#[derive(Clone, Copy, Debug)]
pub struct CoordinationOptions {
    pub cutoff: f64,
    pub kcn: f64,
}

impl Default for CoordinationOptions {
    fn default() -> Self {
        Self {
            cutoff: 25.0,
            kcn: 16.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CoordinationPairDerivative {
    pub i: usize,
    pub j: usize,
    pub r_ij: Vec3,
    pub value: f64,
    pub dcn_dr: f64,
}

#[derive(Clone, Debug)]
pub struct CoordinationDerivatives {
    pub cn: Vec<f64>,
    pub pairs: Vec<CoordinationPairDerivative>,
}

pub fn coordination_numbers(system: &PeriodicSystem) -> Result<Vec<f64>> {
    Ok(coordination_with_derivatives(system, CoordinationOptions::default())?.cn)
}

pub fn coordination_with_derivatives(
    system: &PeriodicSystem,
    options: CoordinationOptions,
) -> Result<CoordinationDerivatives> {
    let nat = system.atoms.len();
    let radii = system
        .atoms
        .iter()
        .map(|atom| covalent_radius_d3_bohr(atom.z))
        .collect::<Result<Vec<_>>>()?;
    let mut cn = vec![0.0; nat];
    let mut pairs = Vec::new();

    for_each_unique_short_range_pair(system, options.cutoff, |pair| {
        let rc = radii[pair.i] + radii[pair.j];
        if rc <= DIST_EPS {
            return Ok(());
        }
        let (value, dcn_dr) = exp_count_value_derivative(options.kcn, pair.r, rc);
        if pair.i == pair.j {
            cn[pair.i] += 2.0 * value;
            pairs.push(CoordinationPairDerivative {
                i: pair.i,
                j: pair.j,
                r_ij: -pair.dr,
                value: 2.0 * value,
                dcn_dr: 2.0 * dcn_dr,
            });
        } else {
            cn[pair.i] += value;
            cn[pair.j] += value;
            pairs.push(CoordinationPairDerivative {
                i: pair.i,
                j: pair.j,
                r_ij: -pair.dr,
                value,
                dcn_dr,
            });
        }
        Ok(())
    })?;

    Ok(CoordinationDerivatives { cn, pairs })
}

pub fn exp_count_value_derivative(kcn: f64, r: f64, rc: f64) -> (f64, f64) {
    let arg = (-kcn * (rc / r - 1.0)).clamp(-80.0, 80.0);
    let expterm = arg.exp();
    let value = 1.0 / (1.0 + expterm);
    let dvalue_dr = if !(-80.0..=80.0).contains(&(-kcn * (rc / r - 1.0))) {
        0.0
    } else {
        -kcn * rc * expterm / (r * r * (1.0 + expterm).powi(2))
    };
    (value, dvalue_dr)
}
