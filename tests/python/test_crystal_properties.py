"""Public crystal property APIs preserve analytic ML physics and explicit units."""
import hashlib
import math
import numpy as np
import pytest
from ase import Atoms
from ase.units import Bohr, Hartree, GPa
from gfn1_rs import Gfn1NativeCalculator, GFN1RSCalculator

NUMBERS=[1,1]
POSITIONS=[[.2,.3,.1],[1.6,.45,.2]]  # bohr
CELL=[[5.6,.3,0.],[0.,6.1,.2],[.1,0.,6.4]]
POSITIONS_A=(np.asarray(POSITIONS)*Bohr).tolist()
CELL_A=(np.asarray(CELL)*Bohr).tolist()
SETTINGS=dict(electronic_temperature=300., energy_tolerance=1e-13,
              charge_tolerance=1e-11,pbc_ao_cutoff=20.,threads=2,memory_mb=512)


@pytest.fixture
def model_file(tmp_path):
    path=tmp_path/'spectral user weights.model'
    path.write_text('''gfn1-rs-radial-hamiltonian-v1
units hartree bohr
species 1
centers 1.2 3.5 5.8
width 1
cutoff 9
limit 0.05
license GPL-3.0-or-later
provenance Independent nonzero structural-head regression; no accuracy claim
head 1 0 0 0.003 -0.0025 0.004
head 1 1 0 0 0 0
head 1 2 0 0 0 0
head 1 3 0.002 0.0004 0.0002 -0.0003
''')
    return path


def atoms_with_calculator(**options):
    atoms=Atoms(numbers=NUMBERS,positions=np.asarray(POSITIONS)*Bohr,
                cell=np.asarray(CELL)*Bohr,pbc=True,masses=[1.,1.])
    atoms.calc=GFN1RSCalculator(**SETTINGS,**options)
    return atoms


def test_harmonic_dos_units_and_invalid_modes():
    calc=Gfn1NativeCalculator(**SETTINGS)
    rows=[[0.,1000.,2000.]]
    native=calc.harmonic_thermodynamics(rows,[1.],[0.,300.,1e8])
    ase=GFN1RSCalculator(**SETTINGS).get_harmonic_thermodynamics(rows,[1.],[0.,300.,1e8])
    assert native[0]['entropy_hartree_per_kelvin']==0.
    for raw,converted in zip(native,ase):
        assert converted['free_energy']==pytest.approx(raw['free_energy_hartree']*Hartree,abs=1e-12)
        assert converted['entropy']==pytest.approx(raw['entropy_hartree_per_kelvin']*Hartree,abs=1e-15)
    # The model keeps the SI/CODATA shared kB; ASE converts the energy unit.
    expected_kb=(1.380649e-23/4.3597447222071e-18)*Hartree
    assert ase[-1]['heat_capacity']==pytest.approx(2*expected_kb,rel=2e-9)
    assert native[0]['excluded_zero_modes_per_primitive_cell']==1.
    with pytest.raises(ValueError,match='imaginary mode'):
        calc.harmonic_thermodynamics([[-10.,1000.]],[1.],[300.])
    grid=np.linspace(-100,2200,2301).tolist()
    dos=calc.phonon_dos(rows,[1.],grid,sigma_cm1=5.)
    assert dos['integrated_states']==pytest.approx(3.,abs=1e-12)
    with pytest.raises(ValueError,match='sum to one'):
        calc.phonon_dos(rows,[.5],grid)


def test_same_eV_electronic_dos_path_without_unit_roundtrip():
    grid=np.linspace(-3,3,3001).tolist()
    calc=Gfn1NativeCalculator(**SETTINGS)
    native=calc.electronic_dos([[-1.,1.]],[1.],grid,sigma_ev=.05,energy_unit='ev')
    ase=GFN1RSCalculator(**SETTINGS).get_electronic_dos([[-1.,1.]],[1.],grid,sigma_ev=.05)
    np.testing.assert_array_equal(ase['density'],native['density'])
    assert native['expected_states']==4.
    assert native['integrated_states']==pytest.approx(4.,abs=1e-12)
    assert native['density'][1000]==pytest.approx(2/(.05*math.sqrt(2*math.pi)),rel=1e-12)


def test_ml_band_and_mesh_retain_user_weight_identity(model_file):
    calc=Gfn1NativeCalculator(**SETTINGS,ml_model_path=str(model_file))
    identity=calc.ml_model_info()
    bands=calc.band_structure(NUMBERS,POSITIONS_A,CELL_A,[(0.,0.,0.),(.2,.1,0.)],kmesh=(2,1,1),gamma_centered=True)
    assert bands['ml_model_sha256']==identity['canonical_sha256']
    assert bands['ml_model_identity']['source_sha256']==hashlib.sha256(model_file.read_bytes()).hexdigest()
    assert bands['electronic_temperature_kelvin']==300.
    plain=Gfn1NativeCalculator(**SETTINGS).band_structure(NUMBERS,POSITIONS_A,CELL_A,[(0.,0.,0.),(.2,.1,0.)],kmesh=(2,1,1),gamma_centered=True)
    assert np.max(np.abs(np.asarray(bands['energies_hartree'])-plain['energies_hartree']))>1e-5
    mesh=calc.phonon_mesh(NUMBERS,POSITIONS_A,CELL_A,(2,1,1),masses_amu=[1.,1.])
    assert mesh['ml_model_sha256']==identity['canonical_sha256']
    assert mesh['weights']==[.5,.5]
    assert np.asarray(mesh['wavenumbers_cm1']).shape==(2,6)
    assert 'Gamma of supercell' in mesh['electronic_sampling']
    assert np.isfinite(mesh['reference_energy_hartree_per_primitive_cell'])
    atoms=atoms_with_calculator(ml_model_path=str(model_file))
    ase_mesh=atoms.calc.get_phonon_mesh((2,1,1),atoms=atoms)
    np.testing.assert_allclose(ase_mesh['wavenumbers_cm1'],mesh['wavenumbers_cm1'],atol=5e-4,rtol=0)
    assert ase_mesh['reference_energy_per_primitive_cell']==pytest.approx(mesh['reference_energy_hartree_per_primitive_cell']*Hartree,abs=1e-10)
    ase_bands=atoms.calc.get_band_structure([(0.,0.,0.),(.2,.1,0.)],atoms=atoms,kmesh=(2,1,1),gamma_centered=True)
    np.testing.assert_allclose(ase_bands['energies'],np.asarray(bands['energies_hartree'])*Hartree,atol=1e-10,rtol=0)


def test_crystal_mesh_guards_and_configured_memory():
    atoms=atoms_with_calculator(kgrid=(2,1,1))
    with pytest.raises(ValueError,match='Gamma of its supercell'):
        atoms.calc.get_phonon_mesh((2,1,1),atoms=atoms)
    low=dict(SETTINGS);low['memory_mb']=1
    with pytest.raises(ValueError,match='budget'):
        Gfn1NativeCalculator(**low).phonon_mesh(NUMBERS,POSITIONS,CELL,(2,2,2),unit='bohr')
    with pytest.raises(ValueError,match='kmesh dimensions'):
        Gfn1NativeCalculator(**SETTINGS).band_structure(NUMBERS,POSITIONS,CELL,[(0.,0.,0.)],unit='bohr',kmesh=(0,1,1))


def test_polar_and_directional_lo_to_postprocessors_preserve_units_and_charge_sum():
    native=Gfn1NativeCalculator(**SETTINGS)
    ase=GFN1RSCalculator(**SETTINGS)
    spring=.002
    h=spring*np.block([[np.eye(3),-np.eye(3)],[-np.eye(3),np.eye(3)]])
    z=np.asarray([.8*np.eye(3),-.8*np.eye(3)])
    masses=[2.,3.]; volume=100.; eps=np.diag([2.,3.,4.])
    raw=native.polar_mode_properties(h.tolist(),z.tolist(),masses,volume)
    converted=ase.get_polar_mode_properties(h*Hartree/Bohr**2,z,masses,volume*Bohr**3)
    np.testing.assert_allclose(raw['ionic_dielectric'],np.eye(3)*4*math.pi*.8**2/(volume*spring),atol=1e-11,rtol=0)
    np.testing.assert_allclose(converted['wavenumbers_cm1'],raw['wavenumbers_cm1'],atol=1e-5,rtol=0)
    np.testing.assert_allclose(converted['mode_effective_charges'],raw['mode_effective_charges_e_per_sqrt_amu'],atol=1e-12,rtol=0)
    raw_lo=native.lo_to_modes(h.tolist(),z.tolist(),eps.tolist(),volume,masses,(1.,0.,0.))
    ase_lo=ase.get_lo_to_modes(h*Hartree/Bohr**2,z,eps,volume*Bohr**3,masses,(1.,0.,0.))
    nac=np.asarray(raw_lo['nonanalytic_correction_hartree_per_bohr2_amu'])
    expected_trace=4*math.pi*.8**2/(volume*2.)*(1/masses[0]+1/masses[1])
    assert np.trace(nac)==pytest.approx(expected_trace,rel=1e-12)
    np.testing.assert_allclose(ase_lo['nonanalytic_correction_per_amu'],nac*Hartree/Bohr**2,atol=1e-13,rtol=0)
    np.testing.assert_allclose(ase_lo['wavenumbers_cm1'],raw_lo['wavenumbers_cm1'],atol=1e-5,rtol=0)
    with pytest.raises(ValueError,match='positive definite'):
        ase.get_lo_to_modes(h*Hartree/Bohr**2,z,-eps,volume*Bohr**3,masses,(1.,0.,0.))


def test_ml_bulk_response_three_api_boundaries_and_unavailable_relaxation(model_file):
    native=Gfn1NativeCalculator(**SETTINGS,ml_model_path=str(model_file))
    identity=native.ml_model_info()
    atoms=atoms_with_calculator(ml_model_path=str(model_file),kgrid=(2,1,1))
    bulk=native.electromechanical_response(NUMBERS,POSITIONS_A,CELL_A,kmesh=(2,1,1),masses_amu=[1.,1.])
    converted=atoms.calc.get_electromechanical_response(atoms=atoms)
    assert bulk['electronic']['electronic_temperature_kelvin']==300.
    assert bulk['berry']['ml_model_sha256']==identity['canonical_sha256']
    assert bulk['electronic']['ml_model_sha256']==identity['canonical_sha256']
    np.testing.assert_allclose(converted['berry']['born_effective_charges'],bulk['berry']['born_effective_charges'],atol=1e-10,rtol=0)
    np.testing.assert_allclose(converted['berry']['clamped_piezoelectric'],np.asarray(bulk['berry']['clamped_piezoelectric_au'])/Bohr**2,atol=1e-10,rtol=0)
    np.testing.assert_allclose(converted['elastic']['nuclear_hessian'],np.asarray(bulk['elastic']['nuclear_hessian_hartree_per_bohr2'])*Hartree/Bohr**2,atol=1e-10,rtol=0)
    assert bulk['relaxed_piezoelectric_au'] is None
    assert converted['relaxed_piezoelectric'] is None
    assert converted['piezoelectric_strain'] is None
    assert 'gradient' in converted['relaxed_response_error']
    born=atoms.calc.get_born_effective_charges(atoms=atoms)
    eps=atoms.calc.get_dielectric_tensor(atoms=atoms)
    polarization=atoms.calc.get_berry_polarization(atoms=atoms)
    np.testing.assert_allclose(born['born_effective_charges'],bulk['berry']['born_effective_charges'],atol=1e-10,rtol=0)
    np.testing.assert_allclose(eps['electronic_relative_dielectric'],bulk['electronic']['electronic_relative_dielectric'],atol=1e-10,rtol=0)
    np.testing.assert_allclose(polarization['polarization'],np.asarray(polarization['polarization_au'])/Bohr**2,atol=1e-12,rtol=0)
    assert polarization['mesh']==[2,1,1]


def test_property_fine_tuning_warm_start_analytic_labels_and_saved_identity(model_file,tmp_path):
    # Known changed teacher weights are an implementation test, not chemical data.
    teacher_path=tmp_path/'property teacher.model'
    text=model_file.read_text().replace('head 1 3 0.002 0.0004 0.0002 -0.0003',
        'head 1 3 0.002 0.005 0.0002 -0.0003')
    teacher_path.write_text(text)
    teacher=Gfn1NativeCalculator(**SETTINGS,ml_model_path=str(teacher_path))
    samples=[]
    for split,r in [('train',.67),('train',.92),('validation',.79)]:
        pos=[[0.,0.,0.],[r,0.,0.]]
        result=teacher.calculate(NUMBERS,pos,compute_gradient=True)
        samples.append(dict(id=f'{split}-{r}',split=split,charge=0,electronic_temperature_kelvin=300.,
            xyz=f'2\nsynthetic ML teacher\nH 0 0 0\nH {r} 0 0\n',targets=[
                dict(observable='energy',unit='hartree',values=result.energy_hartree,scale=.1),
                dict(observable='forces',unit='hartree/bohr',values=(-np.asarray(result.gradient_hartree_per_bohr)).tolist(),scale=.1),
                dict(observable='hessian',unit='hartree/bohr^2',values=teacher.hessian(NUMBERS,pos),scale=.1)]))
    data=dict(schema='gfn1-rs-property-training-v1',coordinate_units='angstrom',
        provenance='Known nonzero synthetic ML property labels',license_spdx='GPL-3.0-or-later',
        reference_method='synthetic native ML Hamiltonian',independent_reference=False,samples=samples)
    settings=dict(parameters=['1:correction:1'],max_epochs=25,learning_rate=.1,gradient_tolerance=1e-10,loss_tolerance=1e-18)
    calc=Gfn1NativeCalculator(**SETTINGS,ml_model_path=str(model_file))
    original=calc.ml_model_info()['canonical_sha256']
    output=tmp_path/'native trained.model'
    report=calc.fine_tune_properties(data,settings,output_model=str(output))
    assert calc.ml_model_info()['canonical_sha256']==original
    assert report['final_train']['loss']<report['initial_train']['loss']*1e-4
    assert report['final_validation']['loss']<report['initial_validation']['loss']*1e-4
    loaded=Gfn1NativeCalculator(**SETTINGS,ml_model_path=str(output))
    assert loaded.ml_model_info()['canonical_sha256']==report['final_model_canonical_sha256']
    assert report['saved_model_file_sha256']==hashlib.sha256(output.read_bytes()).hexdigest()
    ase_output=tmp_path/'ase trained.model'
    ase=GFN1RSCalculator(**SETTINGS,ml_model_path=str(model_file))
    ase_report=ase.fine_tune_properties(data,settings,output_model=ase_output)
    assert ase_report['final_model_canonical_sha256']==report['final_model_canonical_sha256']
    assert len(ase.property_training_observables())==17


def test_material_postprocessing_and_eos_unit_conventions():
    calc=Gfn1NativeCalculator(**SETTINGS)
    ase=GFN1RSCalculator(**SETTINGS)
    lam,mu=.004,.003
    stiffness=np.zeros((6,6));stiffness[:3,:3]=lam;np.fill_diagonal(stiffness[:3,:3],lam+2*mu)
    stiffness[3:,3:]=np.eye(3)*mu
    raw=calc.elastic_properties(stiffness.tolist(),3000.)
    converted=ase.get_elastic_properties(stiffness*Hartree/Bohr**3,3000.)
    assert raw['poisson_hill']==pytest.approx(lam/(2*(lam+mu)),abs=1e-12)
    np.testing.assert_allclose(converted['compliance'],np.asarray(raw['compliance_bohr3_per_hartree'])*Bohr**3/Hartree,atol=1e-12,rtol=0)
    assert converted['young_hill']==pytest.approx(raw['young_hill_gpa']*GPa,rel=1e-12)
    directional=calc.directional_elastic_properties(stiffness.tolist(),3000.,(1.,0.,0.),(0.,1.,0.))
    assert directional['young_gpa']==pytest.approx(raw['young_hill_gpa'],rel=1e-12)
    assert directional['sound_speeds_m_per_s'][0]==pytest.approx(raw['transverse_hill_m_per_s'],rel=1e-12)
    volumes=np.linspace(92.,108.,9);f=.5*((100/volumes)**(2/3)-1)
    energies=-2.+4.5*.01*100*f*f
    fit=calc.fit_equation_of_state(volumes.tolist(),energies.tolist())
    assert fit['equilibrium_volume_bohr3']==pytest.approx(100.,rel=1e-10)
    assert fit['bulk_modulus_hartree_per_bohr3']==pytest.approx(.01,rel=1e-9)
    converted=ase.get_fit_equation_of_state(volumes*Bohr**3,energies*Hartree)
    assert converted['equilibrium_volume']==pytest.approx(100*Bohr**3,rel=1e-10)
    assert converted['minimum_energy']==pytest.approx(-2.*Hartree,abs=1e-10)


def test_analytic_ml_elastic_response_and_relaxation_wrappers(model_file):
    calc=Gfn1NativeCalculator(**SETTINGS,ml_model_path=str(model_file))
    raw=calc.elastic_constants(NUMBERS,POSITIONS_A,CELL_A,kmesh=(2,1,1),gamma_centered=True)
    assert raw['method'].startswith('analytic joint')
    assert raw['ml_model_identity']['canonical_sha256']==calc.ml_model_info()['canonical_sha256']
    assert np.asarray(raw['clamped_stiffness_hartree_per_bohr3']).shape==(6,6)
    assert np.max(np.abs(raw['raw_max_asymmetry']))<1e-8
    assert 'relaxed_response_error' in raw and 'reference_stress_hartree_per_bohr3' in raw
    atoms=atoms_with_calculator(ml_model_path=str(model_file));before=atoms.positions.copy()
    converted=atoms.calc.get_elastic_constants(atoms=atoms,kmesh=(2,1,1),gamma_centered=True)
    np.testing.assert_allclose(converted['clamped_stiffness'],np.asarray(raw['clamped_stiffness_hartree_per_bohr3'])*Hartree/Bohr**3,atol=1e-9,rtol=0)
    trial=atoms.calc.get_relax_periodic(atoms=atoms,max_iterations=1,max_atom_step=.01)
    assert trial['iterations']<=1
    np.testing.assert_array_equal(atoms.positions,before)
    np.testing.assert_allclose(trial['positions'],np.asarray(trial['positions_bohr'])*Bohr,atol=0,rtol=0)


def test_qha_native_and_ase_zero_temperature_and_units():
    volumes=np.linspace(92.,108.,9);f=.5*((100/volumes)**(2/3)-1)
    energies=-2.+4.5*.01*100*f*f
    phonons=[[[1000*(v/100)**-1.2,1500*(v/100)**-1.2,2000*(v/100)**-1.2]] for v in volumes]
    calc=Gfn1NativeCalculator(**SETTINGS)
    raw=calc.quasiharmonic_zero_pressure(volumes.tolist(),energies.tolist(),phonons,[1.],[0.,300.])
    converted=GFN1RSCalculator(**SETTINGS).get_quasiharmonic_zero_pressure(volumes*Bohr**3,energies*Hartree,phonons,[1.],[0.,300.])
    assert raw[0]['thermal_expansion_per_kelvin']==0.
    assert raw[0]['heat_capacity_cp_hartree_per_kelvin']==0.
    for r,c in zip(raw,converted):
        assert c['equilibrium_volume']==pytest.approx(r['equilibrium_volume_bohr3']*Bohr**3,rel=1e-10)
        assert c['heat_capacity_cp']==pytest.approx(r['heat_capacity_cp_hartree_per_kelvin']*Hartree,abs=1e-13)
        assert c['heat_capacity_cp']>=c['heat_capacity_cv']


def test_actual_equation_of_state_driver_around_verified_model_diamond_minimum():
    # This is a coarse Gamma model minimum obtained from analytic-stress cell
    # relaxation. It is deliberately not labelled an experimental lattice.
    volume=145.12572651736696
    a=(4*volume)**(1/3)
    cell=[[0.,a/2,a/2],[a/2,0.,a/2],[a/2,a/2,0.]]
    positions=[[0.,0.,0.],[a/4,a/4,a/4]]
    options=dict(SETTINGS);options['pbc_ao_cutoff']=30.
    calc=Gfn1NativeCalculator(**options)
    result=calc.equation_of_state([6,6],positions,cell,np.linspace(.97,1.03,7).tolist(),
                                  unit='bohr',kmesh=(1,1,1),gamma_centered=True)
    assert result['electronic_temperature_kelvin']==300.
    assert len(result['energies_hartree'])==7
    fit=result['fit']
    assert .97*volume<fit['equilibrium_volume_bohr3']<1.03*volume
    assert fit['rms_energy_error_hartree']<2e-6
    scale=(fit['equilibrium_volume_bohr3']/volume)**(1/3)
    independent=calc.calculate_periodic([6,6],(np.asarray(positions)*scale).tolist(),
        (np.asarray(cell)*scale).tolist(),unit='bohr',kgrid=(1,1,1),gamma_centered=True,compute_stress=True)
    pressure_ev_per_a3=-np.trace(np.asarray(independent.stress))*Hartree/Bohr**3/3
    # Finite-window BM3 fit stationarity is approximate; retain its fit error.
    assert abs(pressure_ev_per_a3/GPa)<.2


def test_native_bohr_and_angstrom_inputs_use_the_native_length_conversion():
    from gfn1_rs.native import BOHR_TO_ANGSTROM
    calc=Gfn1NativeCalculator(**SETTINGS)
    path=[(0.,0.,0.),(.2,.1,0.)]
    atomic=calc.band_structure(NUMBERS,POSITIONS,CELL,path,unit='bohr')
    converted=calc.band_structure(NUMBERS,(np.asarray(POSITIONS)*BOHR_TO_ANGSTROM).tolist(),
        (np.asarray(CELL)*BOHR_TO_ANGSTROM).tolist(),path,unit='angstrom')
    np.testing.assert_allclose(atomic['energies_hartree'],converted['energies_hartree'],atol=1e-12,rtol=0)
