use gfn1_rs::pbc::electromechanical::{strain_skeleton_at_k, strain_skeletons};
use gfn1_rs::pbc::hessian::KpointSkeleton;
use gfn1_rs::pbc::{run_pbc_scc, KMesh, PbcOptions};
use gfn1_rs::{ElectronicOptions, Gfn1Parameters, PeriodicSystem};
use std::time::Instant;

fn compare(a: &[KpointSkeleton], b: &[KpointSkeleton]) -> f64 {
    let mut error = 0.0_f64;
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert_eq!(a.coordination, b.coordination);
        assert_eq!(a.shell_potential, b.shell_potential);
        for (x, y) in a
            .overlap
            .iter()
            .chain(&a.fock)
            .zip(b.overlap.iter().chain(&b.fock))
        {
            error = error
                .max(x.re.max_abs_diff(&y.re))
                .max(x.im.max_abs_diff(&y.im));
        }
    }
    error
}
fn main() -> gfn1_rs::Result<()> {
    let context = gfn1_rs::resources::ResourceContext::new(
        gfn1_rs::resources::ResourceLimits::from_megabytes(Some(2), Some(512))?,
    )?;
    context.run(||{
        let system=PeriodicSystem::from_xyz_str("2\nLattice=\"0 2.18 2.18 2.18 0 2.18 2.18 2.18 0\" pbc=\"T T T\"\nSi 0 0 0\nC 1.09 1.09 1.09\n",0.,false)?;
        let params=Gfn1Parameters::builtin()?;
        let mut model=gfn1_rs::ml::MlHamiltonian::zero(vec![6,14],vec![2.,4.,6.],9.);
        for z in [6,14] {
            let head=model.heads.get_mut(&z).unwrap();
            for level in &mut head.levels {level.weights=vec![0.0003,-0.0002,0.0001,-0.0001,0.00015,-0.00005];}
            head.correction.weights=vec![0.0001,0.0002,-0.0001,-0.0002,0.0001,-0.00005];
        }
        let options=ElectronicOptions{electronic_temperature:300.,energy_tolerance:1e-13,charge_tolerance:1e-11,
            ml_model:Some(std::sync::Arc::new(model)),..Default::default()};
        let mut measurements=Vec::new();
        for mesh in [[2,2,2],[4,4,4]] {
        let pbc=PbcOptions{kmesh:KMesh{size:mesh,gamma_centered:true,fold_time_reversal:true},ao_cutoff:20.,..Default::default()};
        let state=run_pbc_scc(&system,&params,&options,&pbc)?;
        let legacy=||state.kpoints.iter().map(|point|strain_skeleton_at_k(&system,&params,&state,&options,&pbc,point.fractional)).collect::<gfn1_rs::Result<Vec<_>>>();
        let bulk=||strain_skeletons(&system,&params,&state,&options,&pbc);
        let mut old_times=Vec::new();let mut new_times=Vec::new();let mut errors=Vec::new();
        for repeat in 0..3 {
            let (baseline,result,old_time,new_time)=if repeat%2==0 {
                let start=Instant::now();let baseline=legacy()?;let old=start.elapsed().as_secs_f64();
                let start=Instant::now();let result=bulk()?;let new=start.elapsed().as_secs_f64();
                (baseline,result,old,new)
            } else {
                let start=Instant::now();let result=bulk()?;let new=start.elapsed().as_secs_f64();
                let start=Instant::now();let baseline=legacy()?;let old=start.elapsed().as_secs_f64();
                (baseline,result,old,new)
            };
            let error=compare(&baseline,&result);assert_eq!(error,0.,"per-k summation order changed");
            old_times.push(old_time);new_times.push(new_time);errors.push(error);
        }
        let mut sorted_old=old_times.clone();let mut sorted_new=new_times.clone();sorted_old.sort_by(f64::total_cmp);sorted_new.sort_by(f64::total_cmp);
        measurements.push(format!("{{\"mesh\":{mesh:?},\"retained_kpoints\":{},\"ao_basis_size\":{},\"baseline_single_k_loop_seconds\":{old_times:?},\"bulk_shared_geometry_seconds\":{new_times:?},\"baseline_median_seconds\":{},\"bulk_median_seconds\":{},\"median_ratio\":{},\"max_abs_entry_difference\":{errors:?}}}",state.kpoints.len(),state.basis.len(),sorted_old[1],sorted_new[1],sorted_old[1]/sorted_new[1]));
        }
        println!("{{\"schema\":\"gfn1-rs-strain-skeleton-matched-benchmark-v1\",\"electronic_temperature_kelvin\":300,\"threads\":2,\"memory_budget_bytes\":536870912,\"ao_cutoff_bohr\":20,\"nonzero_structural_ml\":true,\"description\":\"Same SCC state for each mesh; three paired repetitions with alternating timing order. Baseline repeats the public single-k wrapper as the previous bulk pattern did. Every real/imaginary overlap/Fock derivative and coordination/shell-potential entry compared exactly. SCC time is excluded; assembly-only ratios are not whole-calculation speedups.\",\"measurements\":[{}]}}",measurements.join(","));
        Ok(())
    })
}
