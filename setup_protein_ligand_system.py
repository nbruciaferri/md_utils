# Imports from the Python standard library
import os
import logging
from sys import stdout
from typing import Iterable

# Imports from the comp chem ecosystem
import openmm.app as app
import numpy as np
import openmm
from openff.units import Quantity, unit
from openmm import unit as openmmunit
from pdbfixer import PDBFixer
from openmmforcefields.generators import EspalomaTemplateGenerator

# Imports from the toolkit
from openff.toolkit import Molecule, Topology
from openff.units.openmm import to_openmm as offquantity_to_openmm

# Imports from RDKit
from rdkit import Chem

def prepare(receptor_path, ligand_path, forcefield):
    receptor = PDBFixer(receptor_path)
    todelete = []
    modeller = app.Modeller(receptor.topology, receptor.positions)
    for atom in modeller.topology.atoms():
        if "H" in atom.name: todelete.append(atom)

    modeller.delete(todelete)

    receptor.topology = modeller.topology
    receptor.positions = modeller.positions
    receptor.findMissingResidues()

    chains = list(receptor.topology.chains())
    keys = receptor.missingResidues.keys()
    for key in list(keys):
        chain = chains[key[0]]
        if key[1] == 0 or key[1] == len(list(chain.residues())):
            del receptor.missingResidues[key]

    receptor.removeHeterogens(keepWater=True)

    receptor.findMissingAtoms() 
    receptor.addMissingAtoms()
    receptor.addMissingHydrogens(7)
    modeller = app.Modeller(receptor.topology, receptor.positions)
    
    if ligand_path is not None:
        ligand = prepare_ligand(ligand_path)
        ligand_topology, ligand_positions = parametrize_ligand(ligand, forcefield=forcefield)
        modeller.add(ligand_topology, ligand_positions)

    modeller.addSolvent(forcefield, ionicStrength=0.15*openmmunit.molar, neutralize=True, 
                    boxShape='cube', padding=1.5*openmmunit.nanometer)
    return modeller

def prepare_ligand(lig_sdf, allow_undefined_stereo=True):
     # Load ligand SDF
    rdkit_mol = Chem.SDMolSupplier(lig_sdf)[0]

    # Convert to OpenMM molecule
    ligand = Molecule.from_rdkit(rdkit_mol,
        allow_undefined_stereo = allow_undefined_stereo
    )
    return ligand

def parametrize_ligand(ligand, forcefield, lig_ff:str='espaloma'):
    
    if lig_ff=='espaloma':
        from openmmforcefields.generators import EspalomaTemplateGenerator
        template_generator =  EspalomaTemplateGenerator(molecules=ligand, forcefield='espaloma-0.3.1')
    elif lig_ff=='SMIRNOFF':
        from openmmforcefields.generators import SMIRNOFFTemplateGenerator
        template_generator =  SMIRNOFFTemplateGenerator(molecules=ligand, forcefield='openff-1.2.0')
    elif lig_ff == 'GAFF':
        from openmmforcefields.generators import GAFFTemplateGenerator
        template_generator = GAFFTemplateGenerator(molecules=ligand, forcefield='gaff-2.11')
    else:
        logging.error(f'Ligand forcefield must be one of espaloma, SMIRNOFF or GAFF')
    
    # add in the Espaloma template generator
    forcefield.registerTemplateGenerator(template_generator.generator)
    
    # make an OpenFF Topology of the ligand
    ligand_off_topology = Topology.from_molecules(molecules=[ligand])

    # convert it to an OpenMM Topology
    ligand_omm_topology = ligand_off_topology.to_openmm()

    # get the positions of the ligand
    ligand_positions = offquantity_to_openmm(ligand.conformers[0])

    return ligand_omm_topology, ligand_positions

def run_simulation(receptor_code: str = None,
                   ligand_code: str = None, 
                   warming_steps: int = 100000,
                   simulation_steps: int = 25000000,
                   seed: int = None):
    receptor_path = f"/mnt/bigdisk1/rotakit_validation/receptors/{receptor_code}.pdb"
    results_path = os.path.join("/mnt/bigdisk1/rotakit_validation/simulations", f"{receptor_code}", f"{ligand_code}", "output")
    ligand_path = None
    
    os.makedirs(results_path, exist_ok=True)

    # Temperature annealing
    Tstart = 5
    Tend = 300
    Tstep = 5
    
    total_steps = warming_steps + simulation_steps

    forcefield = app.ForceField('amber14/protein.ff14SB.xml', 'amber14/tip3pfb.xml', 'amber/tip3p_HFE_multivalent.xml')

    modeller = prepare(receptor_path, ligand_path, forcefield)

    system = forcefield.createSystem(modeller.topology,
                                     nonbondedMethod=app.PME,
                                     nonbondedCutoff=10*openmmunit.angstrom,
                                     switchDistance=9*openmmunit.angstrom,
                                     removeCMMotion=True,
                                     constraints=app.HBonds,
                                     hydrogenMass=3.0*openmmunit.amu)
    
    app.PDBFile.writeFile(modeller.topology,
                          modeller.positions,
                          open(os.path.join(results_path, "system.pdb"), "w"))

    print('Selecting simulation platform')
    try:
        platform = openmm.Platform.getPlatformByName("CUDA")
        platform.setPropertyDefaultValue('DeterministicForces', 'true')
        platform.setPropertyDefaultValue('CudaPrecision', 'mixed')
        platform.setPropertyDefaultValue('CudaDeviceIndex', '0')
        print('Using GPU:CUDA')
    except: 
        try:
            platform = openmm.Platform.getPlatformByName("OpenCL")
            platform.setPropertyDefaultValue('DeterministicForces', 'true')
            platform.setPropertyDefaultValue('Precision', 'mixed')
            print('Using GPU:OpenCL')
        except:
            platform = openmm.Platform.getPlatformByName("CPU")
            print("Switching to CPU, no GPU available.")

    integrator = openmm.LangevinMiddleIntegrator(Tstart * openmmunit.kelvin,
                                          1 / openmmunit.picosecond,
                                          0.001 * openmmunit.picosecond)
    if seed is not None:
        integrator.setRandomNumberSeed(seed)
    
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
        
    print('Adding reporters to the simulation')
    #every 0.1ns
    simulation.reporters.append(app.StateDataReporter(os.path.join(results_path, "statistics.csv"), 25000, step=True, time=True,
                                                totalEnergy=True, potentialEnergy=True, kineticEnergy=True, 
                                                temperature=True, volume=True, density=True,
                                                progress=True, remainingTime=True, speed=True, totalSteps=total_steps))
    #every 0.1ns
    simulation.reporters.append(app.StateDataReporter(stdout, 25000, step=True, time=True,
                                                totalEnergy=True, potentialEnergy=True, kineticEnergy=True, 
                                                temperature=True, volume=True, density=True,
                                                progress=True, remainingTime=True, speed=True, totalSteps=total_steps, separator='\t'))
    
    #every 0.1ns
    simulation.reporters.append(app.DCDReporter(os.path.join(results_path, "trajectory.dcd"),
                                            reportInterval=25000, enforcePeriodicBox=None))

    
    #every 1ns
    simulation.reporters.append(app.CheckpointReporter(os.path.join(results_path,"simualtion.chk"), 250000)) 

    print("Setting positions for the simulation")
    simulation.context.setPositions(modeller.positions)

    print("Minimizing system's energy")
    simulation.minimizeEnergy()

    print(f'Heating system in NVT ensemble for {warming_steps*0.001/1000} ns')
    # Calculate the number of temperature steps
    nT = int((Tend - Tstart) / Tstep)

    # Set initial velocities and temperature
    simulation.context.setVelocitiesToTemperature(Tstart)
    
    # Warm up the system gradually
    for i in range(nT):
        temperature = Tstart + i * Tstep
        integrator.setTemperature(temperature)
        print(f"Temperature set to {temperature} K.")
        simulation.step(int(warming_steps / nT))

    # Increase the timestep for production simulations
    integrator.setStepSize(0.004 * openmmunit.picoseconds)

    print(f'Adding a Montecarlo Barostat to the system')
    system.addForce(openmm.MonteCarloBarostat(1 * openmmunit.bar, Tend * openmmunit.kelvin))
    simulation.context.reinitialize(preserveState=True)

    print(f"Running simulation in NPT ensemble for {simulation_steps*0.004/1000} ns")
    simulation.step(simulation_steps) #25000000 = 100ns
    return

if __name__ == "__main__":
    # rec_lig = {"1cki": "4kbc",
    #            "1cki": None,
    #            "4kbc": "4kbc",
    #            "4kbc": None,
    #            "4b6e": "4b76",
    #            "4b6e": None,
    #            "4b76": "ab76",
    #            "4b76": None,
    #            "5hag": "6fdu",
    #            "5hag": None,
    #            "6fdu": "6fdu",
    #            "6fdu": None}

    rec_lig = {"4kbc": None}
    
    for rec, lig in rec_lig.items():
        print(f"Running {rec}-{lig}")
        run_simulation(receptor_code=rec, ligand_code=lig)