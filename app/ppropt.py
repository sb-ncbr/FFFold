import argparse
import json
import numba
import numpy as np
from Bio import SeqUtils
from Bio.PDB import Select, PDBIO, PDBParser, Superimposer, NeighborSearch
from dataclasses import dataclass
from os import system, path
from multiprocessing import Process, Manager
from multiprocessing import Pool


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
    constrained_atom_symbols: set
    non_constrained_atom_symbols: set
    constrained_atoms: list


@numba.jit(cache=True, nopython=True, fastmath=True, boundscheck=False, nogil=True)
def numba_dist(optimised_residue, residue):
     distances = np.empty(len(optimised_residue))
     mins = np.empty(len(residue))
     for i,a in enumerate(residue):
         for j,b in enumerate(optimised_residue):
             distances[j] = ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)**(1/2)
         mins[i] = distances.min()
     return mins, mins.min()


def optimise_substructures(res, PRO):
        substructure = Substructure(res, PRO)
        substructure_data_dir = f"{PRO.data_dir}/sub_{res.id[1]}"
        system(f"cd {substructure_data_dir} ;"
               f"ulimit -s unlimited ;"
               f"export OMP_NUM_THREADS=1,1 ;"
               f"export OMP_MAX_ACTIVE_LEVELS=1 ;"
               f"export MKL_NUM_THREADS=1 ;"
               f"xtb substructure.pdb --gfnff --input xtb_settings.inp --opt --alpb water --verbose > xtb_output.txt 2>&1 ; rm gfnff_*")
        if not path.isfile(f"{substructure_data_dir}/xtbopt.pdb"): # second try by L-ANCOPT
            substructure_settings = open(f"{substructure_data_dir}/xtb_settings.inp", "r").read().replace("rf", "lbfgs") # zkusit, zda funguje!
            with open(f"{substructure_data_dir}/xtb_settings.inp", "w") as xtb_settings_file:
                xtb_settings_file.write(substructure_settings)
            system(f"cd {substructure_data_dir} ;"
                   f"ulimit -s unlimited ;"
                   f"export OMP_NUM_THREADS=1,1 ;"
                   f"export OMP_MAX_ACTIVE_LEVELS=1 ;"
                   f"export MKL_NUM_THREADS=1 ;"
                   f"xtb substructure.pdb --gfnff --input xtb_settings.inp --opt --alpb water --verbose > xtb_output.txt 2>&1 ; rm gfnff_*")
        return substructure.update_PDB()


class Substructure:
    def __init__(self,
                 optimised_residue,
                 PRO):
        self.optimised_residue = optimised_residue
        self.PRO = PRO
        self.substructure_data_dir = f"{PRO.data_dir}/sub_{self.optimised_residue.id[1]}"
        self.PDBParser = PDBParser(QUIET=True)
        self.residues = []
        self.constrained_atoms_indices = []
        near_residues = sorted(self.PRO.nearest_residues[self.optimised_residue.id[1]-1])
        counter_atoms = 1  # start from 1 because of xtb countering
        for residue in near_residues:
            mins, total_min = numba_dist(np.array([atom.coord for atom in self.optimised_residue.get_atoms()]), np.array([atom.coord for atom in residue.get_atoms()]))
            if total_min < 5:
                constrained_atoms = []
                non_constrained_atoms_symbols = set()
                for atom_distance, atom in zip(mins, residue.get_atoms()):
                    if atom.name == "CA" or atom_distance > 3:
                        constrained_atoms.append(atom)
                        self.constrained_atoms_indices.append(str(counter_atoms))
                    else:
                        non_constrained_atoms_symbols.add(atom.name)
                    counter_atoms += 1
                self.residues.append(Residue(index=residue.id[1],
                                             constrained_atom_symbols={atom.name for atom in constrained_atoms},
                                             non_constrained_atom_symbols=non_constrained_atoms_symbols,
                                             constrained_atoms=constrained_atoms))
        self.residues_indices = {res.index for res in self.residues}
        system(f"mkdir {self.substructure_data_dir}")
        selector = SelectIndexedResidues()
        selector.indices = set([residue.index for residue in self.residues])
        self.PRO.io.save(f"{self.substructure_data_dir}/substructure.pdb", selector)
        xtb_settings_template = """$constrain
atoms: xxx
force constant=1.0
$end
$opt
engine=rf
$end
"""
        substructure_settings = xtb_settings_template.replace("xxx", ", ".join(self.constrained_atoms_indices))
        with open(f"{self.substructure_data_dir}/xtb_settings.inp", "w") as xtb_settings_file:
            xtb_settings_file.write(substructure_settings)


    def update_PDB(self):
        if path.isfile(f"{self.substructure_data_dir}/xtbopt.pdb"):
            category = "Optimised residue"
            optimised_substructure = self.PDBParser.get_structure("substructure", f"{self.substructure_data_dir}/xtbopt.pdb")[0]
            optimised_substructure_residues = list(list(optimised_substructure.get_chains())[0].get_residues())
            constrained_atoms = []
            for optimised_residue, residue in zip(optimised_substructure_residues, self.residues):
                for atom in optimised_residue.get_atoms():
                    if atom.name in residue.constrained_atom_symbols:
                        constrained_atoms.append(atom)
            sup = Superimposer()
            sup.set_atoms([atom for residue in self.residues for atom in residue.constrained_atoms], constrained_atoms)
            sup.apply(optimised_substructure.get_atoms())
            original_atoms_positions = []
            optimised_atoms_positions = []
            for optimised_residue, residue in zip(optimised_substructure_residues, self.residues):
                if residue.index == self.optimised_residue.id[1]:
                    for atom in optimised_residue.get_atoms():
                        optimised_atoms_positions.append(atom.coord)
                    for atom in self.PRO.structure[int(residue.index)]:
                        original_atoms_positions.append(atom.coord)
            residual_rmsd = np.sqrt(np.sum((np.array(original_atoms_positions) - np.array(optimised_atoms_positions)) ** 2) / len(original_atoms_positions))
            if residual_rmsd > 1:
                category = "Highly optimised residue"
        else:
            category = "Not optimised residue"
            residual_rmsd = None
            optimised_atoms_positions = original_atoms_positions
        log = {"residue index": self.optimised_residue.id[1],
                              "residue name": SeqUtils.IUPACData.protein_letters_3to1[self.optimised_residue.resname.capitalize()],
                              "category": category,
                              "residual_rmsd": residual_rmsd}
        return optimised_atoms_positions, log


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
        self._prepare_directory()
        self._load_molecule()

        with Pool(self.cpu) as p:
            optimisations = p.starmap(optimise_substructures, [(res, self) for res in self.residues])

        for i, res in enumerate(self.structure.get_residues()):
            for atom, coord in zip(res.get_atoms(), optimisations[i][0]):
                atom.set_coord(coord)

        logs = [o[1] for o in optimisations]
        with open(f"{self.data_dir}/residues.logs", "w") as residues_logs:
            residues_logs.write(json.dumps(sorted(logs, key=lambda x: x['residue index']), indent=2))
        self.io.save(f"{self.data_dir}/optimised_PDB/{path.basename(self.PDB_file[:-4])}_optimised.pdb")
        if self.delete_auxiliary_files:
            system(f"for au_file in {self.data_dir}/sub_* ; do rm -fr $au_file ; done &")
        print(f"Structure succesfully optimised.\n")

    def _prepare_directory(self):
        print("\nPreparing a data directory... ", end="")
        if path.exists(self.data_dir):
            exit(f"\n\nError! Directory with name {self.data_dir} exists. "
                 f"Remove existed directory or change --data_dir argument.")
        system(f"mkdir {self.data_dir};"
               f"mkdir {self.data_dir}/inputed_PDB;"
               f"mkdir {self.data_dir}/optimised_PDB;"
               f"cp {self.PDB_file} {self.data_dir}/inputed_PDB")
        print("ok\n")

    def _load_molecule(self):
        print(f"Loading of structure from {self.PDB_file}... ", end="")
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
        kdtree = NeighborSearch(list(self.structure.get_atoms()))
        self.nearest_residues = [set(kdtree.search(residue.center_of_mass(geometric=True), 12, level="R"))
                                 for residue in self.residues]
        print("ok\n")


if __name__ == '__main__':
    args = load_arguments()
    PRO(args.data_dir, args.PDB_file, args.cpu, args.delete_auxiliary_files).optimise()
