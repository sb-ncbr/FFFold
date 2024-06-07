import argparse
import json
from Bio import SeqUtils
from Bio.PDB import Select, PDBIO, PDBParser, Superimposer, NeighborSearch
from dataclasses import dataclass
from os import system, path
from multiprocessing import Process, Lock, Array, Queue
from queue import Empty as QueueEmpty
from math import dist
from glob import glob
from time import sleep
import numpy as np
from numba import jit


def load_arguments():
    print("\nParsing arguments... ", end="")
    parser = argparse.ArgumentParser()
    parser.add_argument('--PDB_file', type=str, required=True,
                        help='PDB file with structure, which should be optimised.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory for saving results.')
    parser.add_argument("--delete_auxiliary_files", help="Auxiliary calculation files can be large. With this argument, "
                                                         "the auxiliary files will be continuously deleted during the calculation.",
                        action="store_true")
    parser.add_argument('--cpu', type=int, required=False, default=1,
                        help='How many CPUs should be used for the calculation.')
    args = parser.parse_args()
    if not path.isfile(args.PDB_file):
        print(f"\nERROR! File {args.PDB_file} does not exist!\n")
        exit()
    if path.exists(args.data_dir):
        exit(f"\n\nError! Directory with name {args.data_dir} exists. "
             f"Remove existed directory or change --data_dir argument.")
    print("ok")
    return args


class SelectIndexedResidues(Select):
    def accept_residue(self, residue):
        if residue.id[1] in self.indices:
            return 1
        else:
            return 0


@dataclass
class Residue:
    index: int
    constrained_atoms: list


@jit(cache=True, nopython=True, fastmath=True, boundscheck=False, nogil=True)
def get_distances(optimised_residue, residue):
     distances = np.empty(len(optimised_residue))
     mins = np.empty(len(residue))
     for i,a in enumerate(residue):
         for j,b in enumerate(optimised_residue):
             distances[j] = ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)**(1/2)
         mins[i] = distances.min()
     return mins, mins.min()


def optimise_substructure(coordinates, is_already_optimised, is_currently_optimised_or_queued, queue, lock, PRO):
    while not all(is_already_optimised):
        try:
            optimised_residue = queue.get(block=False)
        except QueueEmpty:
            sleep(0.001)
        else:
            substructure_residues = []  # all residues in substructure
            optimised_residue_index = optimised_residue.id[1]
            constrained_atom_indices = []  # indices of substructure atoms, which should be constrained during optimisation
            counter_atoms = 1  # start from 1 because of xtb countering
            nearest_residues = sorted(PRO.nearest_residues[optimised_residue_index-1])

            for atom in optimised_residue:
                atom.coord = np.array(coordinates[(atom.serial_number - 1) * 3:(atom.serial_number - 1) * 3 + 3])

            for residue in nearest_residues:  # select substructure residues
                for atom in residue.get_atoms():
                    atom.coord = np.array(coordinates[(atom.serial_number-1)*3:(atom.serial_number-1)*3+3])
                minimum_distances, absolute_min_distance = get_distances(np.array([atom.coord for atom in optimised_residue.get_atoms()]),
                                                                         np.array([atom.coord for atom in residue.get_atoms()]))
                if absolute_min_distance < 6:
                    constrained_atoms = []
                    for atom_distance, atom in zip(minimum_distances, residue.get_atoms()):
                        if atom.name == "CA" or atom_distance > 4:
                            constrained_atoms.append(atom)
                            constrained_atom_indices.append(str(counter_atoms))
                        counter_atoms += 1
                    substructure_residues.append(Residue(index=residue.id[1],
                                                         constrained_atoms=constrained_atoms))
            substructure_data_dir = f"{PRO.data_dir}/sub_{optimised_residue_index}"
            system(f"mkdir {substructure_data_dir}")
            selector = SelectIndexedResidues()
            selector.indices = set([residue.index for residue in substructure_residues])
            PRO.io.save(f"{substructure_data_dir}/substructure.pdb", selector)

            # protonation of broken peptide bonds

            num_of_atoms = counter_atoms - 1
            system(f"cd {substructure_data_dir} ;"
                   f"obabel -h -iPDB -oPDB substructure.pdb > reprotonated_substructure.pdb 2>/dev/null")
            with open(f"{substructure_data_dir}/reprotonated_substructure.pdb") as reprotonated_substructure_file:
                atom_lines = [line for line in reprotonated_substructure_file.readlines() if line[:4] == "ATOM"]
                original_atoms = atom_lines[:num_of_atoms]
                added_atoms = atom_lines[num_of_atoms:]
            with open(f"{substructure_data_dir}/repaired_substructure.pdb", "w") as repaired_substructure_file:
                repaired_substructure_file.write("".join(original_atoms))
                added_hydrogen_indices = []
                hydrogens_counter = num_of_atoms
                for line in added_atoms:
                    res_i = int(line[22:26])
                    if any([dist([float(line[30:38]), float(line[38:46]), float(line[46:54])], PRO.structure[res_i]["C"].coord) < 1.1,
                            dist([float(line[30:38]), float(line[38:46]), float(line[46:54])], PRO.structure[res_i]["N"].coord) < 1.1]):
                        repaired_substructure_file.write(line)
                        hydrogens_counter += 1
                        added_hydrogen_indices.append(hydrogens_counter)
            system(f"cd {substructure_data_dir} ; mv repaired_substructure.pdb substructure.pdb")

            # optimise substructure by xtb
            engine = "rf"
            xtb_settings_template = f"""$constrain
            atoms: xxx
            force constant=1.0
            $end
            $opt
            engine={engine}
            $end
            """
            substructure_settings = xtb_settings_template.replace("xxx", ", ".join(
                constrained_atom_indices) + ", " + ", ".join([str(x) for x in added_hydrogen_indices]))
            with open(f"{substructure_data_dir}/xtb_settings.inp", "w") as xtb_settings_file:
                xtb_settings_file.write(substructure_settings)
            run_xtb = (f"cd {substructure_data_dir} ;"
                       f"ulimit -s unlimited ;"
                       f"export OMP_NUM_THREADS=1,1 ;"
                       f"export OMP_MAX_ACTIVE_LEVELS=1 ;"
                       f"export MKL_NUM_THREADS=1 ;"
                       f"xtb substructure.pdb --gfnff --input xtb_settings.inp --opt tight --alpb water --verbose > xtb_output.txt 2>&1 ; rm gfnff_*")
            system(run_xtb)
            if not path.isfile(f"{substructure_data_dir}/xtbopt.pdb"):  # second try by L-ANCOPT
                engine = "lbfgs"
                substructure_settings = open(f"{substructure_data_dir}/xtb_settings.inp", "r").read().replace("rf", engine)
                with open(f"{substructure_data_dir}/xtb_settings.inp", "w") as xtb_settings_file:
                    xtb_settings_file.write(substructure_settings)
                system(run_xtb)

            if path.isfile(f"{substructure_data_dir}/xtbopt.pdb"):
                category = "Optimised residue"
                optimised_substructure = PDBParser(QUIET=True).get_structure("substructure", f"{substructure_data_dir}/xtbopt.pdb")[0]
                optimised_substructure_residues = list(list(optimised_substructure.get_chains())[0].get_residues())
                or_constrained_atoms = []
                op_constrained_atoms = []
                for or_r, op_r in zip(substructure_residues, optimised_substructure_residues):
                    for constrained_atom in or_r.constrained_atoms:
                        or_constrained_atoms.append(constrained_atom)
                        op_constrained_atoms.append(op_r[constrained_atom.name])
                sup = Superimposer()
                sup.set_atoms(or_constrained_atoms, op_constrained_atoms)
                sup.apply(optimised_substructure.get_atoms())
                for op_res, or_res in zip(optimised_substructure_residues, substructure_residues):
                    for atom, coord in zip(PRO.residues[or_res.index-1].get_atoms(), [a.coord for a in op_res.get_atoms()]):
                        if atom not in constrained_atoms:
                            coordinates[(atom.serial_number-1)*3] = coord[0]
                            coordinates[(atom.serial_number-1)*3+1] = coord[1]
                            coordinates[(atom.serial_number-1)*3+2] = coord[2]

            else:
                category = "Not optimised residue"
                engine = "none"
            log = {"residue index": optimised_residue_index,
                   "residue name": SeqUtils.IUPACData.protein_letters_3to1[optimised_residue.resname.capitalize()],
                   "category": category,
                   "optimisation engine": engine}
            with open(f"{substructure_data_dir}/residue.log", "w") as residue_log:
                residue_log.write(json.dumps(log, indent=2))

            is_already_optimised[optimised_residue_index-1] = True
            is_currently_optimised_or_queued[optimised_residue_index-1] = False

            added = False
            for res in nearest_residues:
                res_i = res.id[1] - 1
                if is_already_optimised[res_i] or is_currently_optimised_or_queued[res_i]:
                    continue
                if all(is_already_optimised[less_flexible_residue_index] for less_flexible_residue_index in PRO.less_flexible_residues[res_i]):
                    with lock:
                        if is_currently_optimised_or_queued[res_i] == False:  # == is right because multiprocessing
                            is_currently_optimised_or_queued[res_i] = True
                            queue.put(res)
                            added = True

            if added is False and queue.empty():
                for res in PRO.residues:
                    res_i = res.id[1] - 1
                    if is_already_optimised[res_i] or is_currently_optimised_or_queued[res_i]:
                        continue
                    if all(is_already_optimised[less_flexible_residue_index] for less_flexible_residue_index in
                           PRO.less_flexible_residues[res_i]):
                        with lock:
                            if is_currently_optimised_or_queued[res_i] == False:  # == is right because multiprocessing
                                is_currently_optimised_or_queued[res_i] = True
                                queue.put(res)


class PRO:
    def __init__(self,
                 data_dir: str,
                 PDB_file: str,
                 cpu: int,
                 delete_auxiliary_files: bool):
        self.data_dir = data_dir
        self.PDB_file = PDB_file
        self.cpu = cpu
        self.delete_auxiliary_files = delete_auxiliary_files

    def optimise(self):
        print(f"Loading of structure from {self.PDB_file}... ", end="")
        self._load_molecule()
        print("ok")

        print("Optimisation... ", end="")
        queue = Queue()
        coordinates = Array("d", [coord for atom in self.structure.get_atoms() for coord in atom.coord], lock=False)
        is_already_optimised = Array("d", [0 for _ in self.residues], lock=False)
        is_currently_optimised_or_queued = Array("d", [0 for _ in self.residues], lock=False)
        lock = Lock()
        for seed in self.seeds:
            queue.put(seed)
            is_currently_optimised_or_queued[seed.id[1]-1] = True
        processes = []
        for _ in range(self.cpu):
            p = Process(target=optimise_substructure, args=(coordinates, is_already_optimised, is_currently_optimised_or_queued, queue, lock, self))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        for a in self.structure.get_atoms():
            a.coord = np.array(coordinates[(a.serial_number-1)*3:(a.serial_number-1)*3+3])
        print("ok")

        print("Storage of the optimised structure... ", end="")
        logs = sorted([json.loads(open(f).read()) for f in glob(f"{self.data_dir}/sub_*/residue.log")],
                      key=lambda x: x['residue index'])
        atom_counter = 0
        for optimised_residue, log in zip(self.residues, logs):
            d = 0
            for optimised_atom in optimised_residue.get_atoms():
                d += dist(optimised_atom.coord, self.original_atoms_positions[atom_counter])**2
                atom_counter += 1
            residual_rmsd = (d / len(list(optimised_residue.get_atoms())))**(1/2)
            log["residual_rmsd"] = residual_rmsd
            if residual_rmsd > 1:
                log["category"] = "Highly optimised residue"
        with open(f"{self.data_dir}/residues.logs", "w") as residues_logs:
            residues_logs.write(json.dumps(logs, indent=2))
        self.io.save(f"{self.data_dir}/optimised_PDB/{path.basename(self.PDB_file[:-4])}_optimised.pdb")
        if self.delete_auxiliary_files:
            system(f"for au_file in {self.data_dir}/sub_* ; do rm -fr $au_file ; done &")
        print("ok\n\n")

    def _load_molecule(self):
        system(f"mkdir {self.data_dir};"
               f"mkdir {self.data_dir}/inputed_PDB;"
               f"mkdir {self.data_dir}/optimised_PDB;"
               f"cp {self.PDB_file} {self.data_dir}/inputed_PDB")
        try:
            structure = PDBParser(QUIET=True).get_structure("structure", self.PDB_file)
            io = PDBIO()
            io.set_structure(structure)
            self.io = io
            self.structure = io.structure[0]["A"]
        except KeyError:
            print(f"\nERROR! PDB file {self.PDB_file} does not contain any structure.\n")
            exit()
        self.residues = list(self.structure.get_residues())
        self.original_atoms_positions = [atom.coord for atom in self.structure.get_atoms()]
        amk_radius = {'ALA': 2.4801,
                      'ARG': 4.8618,
                      'ASN': 3.2237,
                      'ASP': 2.8036,
                      'CYS': 2.5439,
                      'GLN': 3.8456,
                      'GLU': 3.3963,
                      'GLY': 2.1455,
                      'HIS': 3.8376,
                      'ILE': 3.4050,
                      'LEU': 3.5357,
                      'LYS': 4.4521,
                      'MET': 4.1821,
                      'PHE': 4.1170,
                      'PRO': 2.8418,
                      'SER': 2.4997,
                      'THR': 2.7487,
                      'TRP': 4.6836,
                      'TYR': 4.5148,
                      'VAL': 2.9515}
        kdtree = NeighborSearch(list(self.structure.get_atoms()))
        self.nearest_residues = [set(kdtree.search(residue.center_of_mass(geometric=True), amk_radius[residue.resname]+6, level="R"))
                                 for residue in self.residues]
        self.density_of_atoms_around_residues = []
        for residue in self.residues:
            volume_c = ((4 / 3) * 3.14 * ((amk_radius[residue.resname]) + 2) ** 3)
            num_of_atoms_c = len(kdtree.search(residue.center_of_mass(geometric=True), (amk_radius[residue.resname]) + 2, level="A"))
            density_c = num_of_atoms_c/volume_c

            volume_2c = ((4 / 3) * 3.14 * ((amk_radius[residue.resname]) + 10) ** 3)
            num_of_atoms_2c = len(kdtree.search(residue.center_of_mass(geometric=True), (amk_radius[residue.resname]) + 8, level="A"))
            density_2c = num_of_atoms_2c/volume_2c

            self.density_of_atoms_around_residues.append(density_c + density_2c/10)

        self.less_flexible_residues = []
        for res in self.residues:
            less_flexible_residues_than_res = []
            for near_residue in self.nearest_residues[res.id[1] - 1]:
                if res == near_residue:
                    continue
                if self.density_of_atoms_around_residues[near_residue.id[1]-1] > self.density_of_atoms_around_residues[res.id[1]-1]:
                    less_flexible_residues_than_res.append(near_residue.id[1]-1)
            self.less_flexible_residues.append(less_flexible_residues_than_res)
        self.seeds = []
        for res, less_flexible_residues in zip(self.residues, self.less_flexible_residues):
            if not less_flexible_residues:
                self.seeds.append(res)


if __name__ == '__main__':
    args = load_arguments()
    PRO(args.data_dir, args.PDB_file, args.cpu, args.delete_auxiliary_files).optimise()
