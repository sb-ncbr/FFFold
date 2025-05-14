import argparse
import json
from rdkit import Chem
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

class AtomSelector(Select):
    """
    Support class for Biopython.
    After initialization, a set with all full ids of the atoms to be written into the substructure must be stored in self.full_ids.
    """
    def accept_atom(self, atom):
        return int(atom.full_id in self.full_ids)



amk_max_radius = {'MET': 4.582198220688083,
                  'VAL': 3.441906011526445,
                  'ASP': 3.5807236327594176,
                  'LYS': 5.142450211142281,
                  'LEU': 4.174556929072557,
                  'ILE': 3.776764078186954,
                  'HIS': 4.3249529464818215,
                  'PRO': 3.440157836371418,
                  'TRP': 5.299980225372395,
                  'SER': 2.859502908272858,
                  'GLY': 2.377182182239665,
                  'ARG': 5.645318573062258,
                  'GLN': 4.263049856809807,
                  'ALA': 2.695115955275209,
                  'GLU': 4.007705840682104,
                  'PHE': 4.670650293844719,
                  'THR': 3.264775624992186,
                  'TYR': 4.8926122882543535,
                  'ASN': 3.6699696442588423,
                  'CYS': 2.941139934314898}


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
            optimised_residue_index = queue.get(block=False)
        except QueueEmpty:
            sleep(0.001)
        else:

            # creation of substructure
            optimised_residue = PRO.residues[optimised_residue_index]
            substructure_data_dir = f"{PRO.data_dir}/sub_{optimised_residue_index+1}"
            system(f"mkdir {substructure_data_dir}")

            # create and save min_radius and max_radius substructures by biopython
            for residue in PRO.nearest_residues[optimised_residue_index]:
                for atom in residue:
                    atom.coord = np.array(coordinates[(atom.serial_number - 1) * 3:(atom.serial_number - 1) * 3 + 3])


            kdtree = NeighborSearch([atom for residue in PRO.nearest_residues[optimised_residue_index] for atom in residue])


            atoms_in_3A = [atom.full_id[1:] for atom in kdtree.search(center=optimised_residue.center_of_mass(geometric=True),
                                                                      radius=3 + amk_max_radius[optimised_residue.resname],
                                                                      level="A") if atom.name != "CA"] # předělat na set, předělat na uncostrained atoms

            atoms_in_6A = kdtree.search(center=optimised_residue.center_of_mass(geometric=True),
                                             radius=6 + amk_max_radius[optimised_residue.resname],
                                             level="A")
            atoms_in_12A = kdtree.search(center=optimised_residue.center_of_mass(geometric=True),
                                              radius=12 + amk_max_radius[optimised_residue.resname],
                                              level="A")
            selector = AtomSelector()
            selector.full_ids = set([atom.full_id for atom in atoms_in_6A])
            PRO.io.save(file=f"{substructure_data_dir}/atoms_in_6A.pdb",
                         select=selector,
                         preserve_atom_numbering=True)
            selector.full_ids = set([atom.full_id for atom in atoms_in_12A])
            PRO.io.save(file=f"{substructure_data_dir}/atoms_in_12A.pdb",
                         select=selector,
                         preserve_atom_numbering=True)

            # load substructures by RDKit to determine bonds
            mol_min_radius = Chem.MolFromPDBFile(pdbFileName=f"{substructure_data_dir}/atoms_in_6A.pdb",
                                                 removeHs=False,
                                                 sanitize=False)
            mol_min_radius_conformer = mol_min_radius.GetConformer()
            mol_max_radius = Chem.MolFromPDBFile(pdbFileName=f"{substructure_data_dir}/atoms_in_12A.pdb",
                                                 removeHs=False,
                                                 sanitize=False)
            mol_max_radius_conformer = mol_max_radius.GetConformer()

            # dictionaries allow quick and precise matching of atoms from mol_min_radius and mol_max_radius
            mol_min_radius_coord_dict = {}
            for i, mol_min_radius_atom in enumerate(mol_min_radius.GetAtoms()):
                coord = mol_min_radius_conformer.GetAtomPosition(i)
                mol_min_radius_coord_dict[(coord.x, coord.y, coord.z)] = mol_min_radius_atom
            mol_max_radius_coord_dict = {}
            for i, mol_max_radius_atom in enumerate(mol_max_radius.GetAtoms()):
                coord = mol_max_radius_conformer.GetAtomPosition(i)
                mol_max_radius_coord_dict[(coord.x, coord.y, coord.z)] = mol_max_radius_atom

            # find atoms from mol_min_radius with broken bonds
            atoms_with_broken_bonds = []
            for mol_min_radius_atom in mol_min_radius.GetAtoms():
                coord = mol_min_radius_conformer.GetAtomPosition(mol_min_radius_atom.GetIdx())
                mol_max_radius_atom = mol_max_radius_coord_dict[(coord.x, coord.y, coord.z)]
                if len(mol_min_radius_atom.GetNeighbors()) != len(mol_max_radius_atom.GetNeighbors()):
                    atoms_with_broken_bonds.append(mol_max_radius_atom)

            # create a substructure that will have only C-C bonds broken
            carbons_with_broken_bonds_coord = []  # hydrogens will be added only to these carbons
            substructure_coord_dict = mol_min_radius_coord_dict
            while atoms_with_broken_bonds:
                atom_with_broken_bonds = atoms_with_broken_bonds.pop(0)
                bonded_atoms = atom_with_broken_bonds.GetNeighbors()
                for bonded_atom in bonded_atoms:
                    coord = mol_max_radius_conformer.GetAtomPosition(bonded_atom.GetIdx())
                    if (coord.x, coord.y, coord.z) in substructure_coord_dict:
                        continue
                    else:
                        if atom_with_broken_bonds.GetSymbol() == "C" and bonded_atom.GetSymbol() == "C":
                            carbons_with_broken_bonds_coord.append(
                                mol_max_radius_conformer.GetAtomPosition(atom_with_broken_bonds.GetIdx()))
                            continue
                        else:
                            atoms_with_broken_bonds.append(bonded_atom)
                            substructure_coord_dict[(coord.x, coord.y, coord.z)] = bonded_atom

            # create substructure in Biopython library
            # we prefer to use kdtree because serial_id may be discontinuous in some pdbs files
            # for example, a structure with PDB code 107d and its serial numbers 218 and 445
            substructure_atoms = [kdtree.search(center=coord,
                                                     radius=0.1,
                                                     level="A")[0] for coord in substructure_coord_dict.keys()]
            selector.full_ids = set([atom.full_id for atom in substructure_atoms])
            PRO.io.save(file=f"{substructure_data_dir}/substructure.pdb",
                         select=selector,
                         preserve_atom_numbering=True)


            substructure = PDBParser(QUIET=True).get_structure(id="structure",
                                                                file=f"{substructure_data_dir}/substructure.pdb")[0] # smazat chainy! a celkově to projít
            # original_substructure_atoms = [atom.full_id[1:] for atom in substructure.get_atoms()] # přepsat na set # asi není potřeba
            # original_substructure_serial_ids = [atom.serial_number for atom in substructure.get_atoms()] # asi není potreba

            # define constrained atoms
            constrained_atom_indices = []
            or_constrained_atoms = []
            for atom_index, atom in enumerate(substructure.get_atoms(), start=1):
                if atom.full_id[1:] in atoms_in_3A:
                    continue
                constrained_atom_indices.append(str(atom_index))
                or_constrained_atoms.append(atom)


            # add hydrogens to broken C-C bonds by openbabel
            system(f"cd {substructure_data_dir} ; obabel -iPDB -oPDB substructure.pdb -h > readded_hydrogens_substructure.pdb 2>/dev/null")
            with open(f"{substructure_data_dir}/readded_hydrogens_substructure.pdb") as readded_hydrogens_substructure_file:
                atom_lines = [line for line in readded_hydrogens_substructure_file.readlines() if
                              line[:4] in ["ATOM", "HETA"]]
                original_atoms_lines = atom_lines[:len(substructure_atoms)]
                added_hydrogens_lines = atom_lines[len(substructure_atoms):]
            with open(f"{substructure_data_dir}/repaired_substructure.pdb", "w") as repaired_substructure_file:
                added_hydrogen_indices = []
                added_hydrogen_indices_counter = len(substructure_atoms) + 1
                repaired_substructure_file.write("".join(original_atoms_lines))
                for added_hydrogen_line in added_hydrogens_lines:
                    added_hydrogen_coord = (float(added_hydrogen_line[30:38]),
                                            float(added_hydrogen_line[38:46]),
                                            float(added_hydrogen_line[46:54]))
                    if any([dist(added_hydrogen_coord, carbon_coord) < 1.3 for carbon_coord in
                            carbons_with_broken_bonds_coord]):
                        repaired_substructure_file.write(added_hydrogen_line)
                        added_hydrogen_indices.append(str(added_hydrogen_indices_counter))  # added hydrogens should be also constrained
                        added_hydrogen_indices_counter += 1



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
                       f"export OMP_STACKSIZE=5G ; "
                       f"export OMP_NUM_THREADS=1,1 ;"
                       f"export OMP_MAX_ACTIVE_LEVELS=1 ;"
                       f"export MKL_NUM_THREADS=1 ;"
                       f"xtb repaired_substructure.pdb --gfnff --input xtb_settings.inp --opt tight --alpb water --verbose > xtb_output.txt 2>&1 ; rm gfnff_*")
            system(run_xtb)
            if not path.isfile(f"{substructure_data_dir}/xtbopt.pdb"):  # second try by L-ANCOPT
                engine = "lbfgs"
                substructure_settings = open(f"{substructure_data_dir}/xtb_settings.inp", "r").read().replace("rf", engine)
                with open(f"{substructure_data_dir}/xtb_settings.inp", "w") as xtb_settings_file:
                    xtb_settings_file.write(substructure_settings)
                system(run_xtb)

            if path.isfile(f"{substructure_data_dir}/xtbopt.pdb"):
                category = "Optimised residue"

                with open(f"{substructure_data_dir}/xtbopt.pdb", "r") as xtb_output:
                    xtb_output_lines = xtb_output.readlines()
                with open(f"{substructure_data_dir}/xtbopt_without_added_hydrogens.pdb", "w") as xtb_output_repaired:
                    xtb_output_repaired.write("".join(xtb_output_lines[:len(xtb_output_lines) - (len(added_hydrogen_indices) + 1)]))
                optimised_substructure = PDBParser(QUIET=True).get_structure("substructure", f"{substructure_data_dir}/xtbopt_without_added_hydrogens.pdb")[0]
                op_constrained_atoms = []
                for op_atom in optimised_substructure.get_atoms():
                    if op_atom.full_id[1:] not in atoms_in_3A:
                        op_constrained_atoms.append(op_atom)
                sup = Superimposer()
                sup.set_atoms(or_constrained_atoms, op_constrained_atoms)
                sup.apply(optimised_substructure.get_atoms())
                for op_atom, or_atom in zip(optimised_substructure.get_atoms(), substructure.get_atoms()):
                    if op_atom.full_id[1:] in atoms_in_3A:
                        coordinates[(or_atom.serial_number - 1) * 3] = op_atom.coord[0]
                        coordinates[(or_atom.serial_number - 1) * 3 + 1] = op_atom.coord[1]
                        coordinates[(or_atom.serial_number - 1) * 3 + 2] = op_atom.coord[2]


            else:
                category = "Not optimised residue"
                engine = "none"
            log = {"residue index": optimised_residue_index + 1,
                   "residue name": SeqUtils.IUPACData.protein_letters_3to1[optimised_residue.resname.capitalize()],
                   "category": category,
                   "optimisation engine": engine}
            with open(f"{substructure_data_dir}/residue.log", "w") as residue_log:
                residue_log.write(json.dumps(log, indent=2))

            is_already_optimised[optimised_residue_index] = True
            is_currently_optimised_or_queued[optimised_residue_index] = False

            added = False
            for res in PRO.nearest_residues[optimised_residue_index]:
                res_i = res.id[1] - 1
                if is_already_optimised[res_i] or is_currently_optimised_or_queued[res_i]:
                    continue
                if all(is_already_optimised[less_flexible_residue_index] for less_flexible_residue_index in PRO.less_flexible_residues[res_i]):
                    with lock:
                        if is_currently_optimised_or_queued[res_i] == False:  # == is right because multiprocessing
                            is_currently_optimised_or_queued[res_i] = True
                            queue.put(res.id[1]-1)
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
                                queue.put(res.id[1]-1)

# class Substructure:
#     def __init__(self, optimised_residue):
#
#         # creation of substructure
#         optimised_residue = PRO.residues[optimised_residue_index]
#         substructure_data_dir = f"{PRO.data_dir}/sub_{optimised_residue_index + 1}"
#         system(f"mkdir {substructure_data_dir}")
#
#         # create and save min_radius and max_radius substructures by biopython
#         for residue in PRO.nearest_residues[optimised_residue_index]:
#             for atom in residue:
#                 atom.coord = np.array(coordinates[(atom.serial_number - 1) * 3:(atom.serial_number - 1) * 3 + 3])
#
#         kdtree = NeighborSearch([atom for residue in PRO.nearest_residues[optimised_residue_index] for atom in residue])
#
#         atoms_in_3A = [atom.full_id[1:] for atom in
#                        kdtree.search(center=optimised_residue.center_of_mass(geometric=True),
#                                      radius=3 + amk_max_radius[optimised_residue.resname],
#                                      level="A") if atom.name != "CA"]  # předělat na set, předělat na uncostrained atoms
#
#         atoms_in_6A = kdtree.search(center=optimised_residue.center_of_mass(geometric=True),
#                                     radius=6 + amk_max_radius[optimised_residue.resname],
#                                     level="A")
#         atoms_in_12A = kdtree.search(center=optimised_residue.center_of_mass(geometric=True),
#                                      radius=12 + amk_max_radius[optimised_residue.resname],
#                                      level="A")
#         selector = AtomSelector()
#         selector.full_ids = set([atom.full_id for atom in atoms_in_6A])
#         PRO.io.save(file=f"{substructure_data_dir}/atoms_in_6A.pdb",
#                     select=selector,
#                     preserve_atom_numbering=True)
#         selector.full_ids = set([atom.full_id for atom in atoms_in_12A])
#         PRO.io.save(file=f"{substructure_data_dir}/atoms_in_12A.pdb",
#                     select=selector,
#                     preserve_atom_numbering=True)
#
#         # load substructures by RDKit to determine bonds
#         mol_min_radius = Chem.MolFromPDBFile(pdbFileName=f"{substructure_data_dir}/atoms_in_6A.pdb",
#                                              removeHs=False,
#                                              sanitize=False)
#         mol_min_radius_conformer = mol_min_radius.GetConformer()
#         mol_max_radius = Chem.MolFromPDBFile(pdbFileName=f"{substructure_data_dir}/atoms_in_12A.pdb",
#                                              removeHs=False,
#                                              sanitize=False)
#         mol_max_radius_conformer = mol_max_radius.GetConformer()
#
#         # dictionaries allow quick and precise matching of atoms from mol_min_radius and mol_max_radius
#         mol_min_radius_coord_dict = {}
#         for i, mol_min_radius_atom in enumerate(mol_min_radius.GetAtoms()):
#             coord = mol_min_radius_conformer.GetAtomPosition(i)
#             mol_min_radius_coord_dict[(coord.x, coord.y, coord.z)] = mol_min_radius_atom
#         mol_max_radius_coord_dict = {}
#         for i, mol_max_radius_atom in enumerate(mol_max_radius.GetAtoms()):
#             coord = mol_max_radius_conformer.GetAtomPosition(i)
#             mol_max_radius_coord_dict[(coord.x, coord.y, coord.z)] = mol_max_radius_atom
#
#         # find atoms from mol_min_radius with broken bonds
#         atoms_with_broken_bonds = []
#         for mol_min_radius_atom in mol_min_radius.GetAtoms():
#             coord = mol_min_radius_conformer.GetAtomPosition(mol_min_radius_atom.GetIdx())
#             mol_max_radius_atom = mol_max_radius_coord_dict[(coord.x, coord.y, coord.z)]
#             if len(mol_min_radius_atom.GetNeighbors()) != len(mol_max_radius_atom.GetNeighbors()):
#                 atoms_with_broken_bonds.append(mol_max_radius_atom)
#
#         # create a substructure that will have only C-C bonds broken
#         carbons_with_broken_bonds_coord = []  # hydrogens will be added only to these carbons
#         substructure_coord_dict = mol_min_radius_coord_dict
#         while atoms_with_broken_bonds:
#             atom_with_broken_bonds = atoms_with_broken_bonds.pop(0)
#             bonded_atoms = atom_with_broken_bonds.GetNeighbors()
#             for bonded_atom in bonded_atoms:
#                 coord = mol_max_radius_conformer.GetAtomPosition(bonded_atom.GetIdx())
#                 if (coord.x, coord.y, coord.z) in substructure_coord_dict:
#                     continue
#                 else:
#                     if atom_with_broken_bonds.GetSymbol() == "C" and bonded_atom.GetSymbol() == "C":
#                         carbons_with_broken_bonds_coord.append(
#                             mol_max_radius_conformer.GetAtomPosition(atom_with_broken_bonds.GetIdx()))
#                         continue
#                     else:
#                         atoms_with_broken_bonds.append(bonded_atom)
#                         substructure_coord_dict[(coord.x, coord.y, coord.z)] = bonded_atom
#
#         # create substructure in Biopython library
#         # we prefer to use kdtree because serial_id may be discontinuous in some pdbs files
#         # for example, a structure with PDB code 107d and its serial numbers 218 and 445
#         substructure_atoms = [kdtree.search(center=coord,
#                                             radius=0.1,
#                                             level="A")[0] for coord in substructure_coord_dict.keys()]
#         selector.full_ids = set([atom.full_id for atom in substructure_atoms])
#         PRO.io.save(file=f"{substructure_data_dir}/substructure.pdb",
#                     select=selector,
#                     preserve_atom_numbering=True)
#
#         substructure = PDBParser(QUIET=True).get_structure(id="structure",
#                                                            file=f"{substructure_data_dir}/substructure.pdb")[
#             0]  # smazat chainy! a celkově to projít
#         # original_substructure_atoms = [atom.full_id[1:] for atom in substructure.get_atoms()] # přepsat na set # asi není potřeba
#         # original_substructure_serial_ids = [atom.serial_number for atom in substructure.get_atoms()] # asi není potreba
#
#         # define constrained atoms
#         constrained_atom_indices = []
#         or_constrained_atoms = []
#         for atom_index, atom in enumerate(substructure.get_atoms(), start=1):
#             if atom.full_id[1:] in atoms_in_3A:
#                 continue
#             constrained_atom_indices.append(str(atom_index))
#             or_constrained_atoms.append(atom)
#
#         # add hydrogens to broken C-C bonds by openbabel
#         system(
#             f"cd {substructure_data_dir} ; obabel -iPDB -oPDB substructure.pdb -h > readded_hydrogens_substructure.pdb 2>/dev/null")
#         with open(f"{substructure_data_dir}/readded_hydrogens_substructure.pdb") as readded_hydrogens_substructure_file:
#             atom_lines = [line for line in readded_hydrogens_substructure_file.readlines() if
#                           line[:4] in ["ATOM", "HETA"]]
#             original_atoms_lines = atom_lines[:len(substructure_atoms)]
#             added_hydrogens_lines = atom_lines[len(substructure_atoms):]
#         with open(f"{substructure_data_dir}/repaired_substructure.pdb", "w") as repaired_substructure_file:
#             added_hydrogen_indices = []
#             added_hydrogen_indices_counter = len(substructure_atoms) + 1
#             repaired_substructure_file.write("".join(original_atoms_lines))
#             for added_hydrogen_line in added_hydrogens_lines:
#                 added_hydrogen_coord = (float(added_hydrogen_line[30:38]),
#                                         float(added_hydrogen_line[38:46]),
#                                         float(added_hydrogen_line[46:54]))
#                 if any([dist(added_hydrogen_coord, carbon_coord) < 1.3 for carbon_coord in
#                         carbons_with_broken_bonds_coord]):
#                     repaired_substructure_file.write(added_hydrogen_line)
#                     added_hydrogen_indices.append(
#                         str(added_hydrogen_indices_counter))  # added hydrogens should be also constrained
#                     added_hydrogen_indices_counter += 1




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
            is_currently_optimised_or_queued[seed] = True
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
            self.structure = io.structure[0]
        except KeyError:
            exit(f"\nERROR! PDB file {self.PDB_file} does not contain any structure.\n")
        self.residues = list(self.structure.get_residues())
        self.original_atoms_positions = [atom.coord for atom in self.structure.get_atoms()]

        kdtree = NeighborSearch(list(self.structure.get_atoms()))
        self.nearest_residues = [set(kdtree.search(residue.center_of_mass(geometric=True), amk_max_radius[residue.resname]+12, level="R"))
                                 for residue in self.residues]
        self.density_of_atoms_around_residues = []
        for residue in self.residues:
            volume1 = ((4 / 3) * 3.14 * ((amk_max_radius[residue.resname]) + 2) ** 3)
            num_of_atoms1 = len(kdtree.search(residue.center_of_mass(geometric=True), (amk_max_radius[residue.resname]) + 2, level="A"))
            density1 = num_of_atoms1/volume1
            volume2 = ((4 / 3) * 3.14 * ((amk_max_radius[residue.resname]) + 15) ** 3)
            num_of_atoms2 = len(kdtree.search(residue.center_of_mass(geometric=True), (amk_max_radius[residue.resname]) + 15, level="A"))
            density2 = num_of_atoms2/volume2
            self.density_of_atoms_around_residues.append(density1 + density2/100)
        # self.seeds = []
        # for res in self.residues:
        #     for near_residue in self.nearest_residues[res.id[1] - 1]:
        #         if res == near_residue:
        #             continue
        #         if self.density_of_atoms_around_residues[near_residue.id[1]-1] > self.density_of_atoms_around_residues[res.id[1]-1]:
        #             break
        #     else:
        #         self.seeds.append(res.id[1]-1)


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
                self.seeds.append(res.id[1]-1)


if __name__ == '__main__':
    args = load_arguments()
    PRO(args.data_dir, args.PDB_file, args.cpu, args.delete_auxiliary_files).optimise()


# předělat ať to funguje s více chainama
# předělat aby to optimalizovalo celé proteiny, poptat se Radky

#  -91.002527580181
#  -90.966581518985

# implementovat
# 1) iterativnost
# 2) substruktura se vytváří před iteracemi

# 3) zkonvergované rezidua se již neoptimalizují
# 4) není potřeba jít od nejzanořenějšího (asi)