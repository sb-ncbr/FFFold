import numpy as np
from numba.typed import List
from rdkit import Chem
from sklearn.neighbors import KDTree as kdtreen

from .amino_acids_atomic_types import real_ats_types
from .problematic_atom_info import problematic_atom_info


class Molecule:
    def __init__(self,
                 pdb_file: str,
                 pqr_file: str):

        # load charges from propka, sum of charges are used as total charge of molecule
        self.total_chg = round(sum(float(line[55:62]) for line in open(pqr_file, "r").readlines()[:-2]))

        # load molecule by rdkit
        self.rdkit_mol = Chem.MolFromPDBFile(pdb_file,
                                             removeHs=False,
                                             sanitize=False)

        # load atoms and bonds
        self.symbols = [atom.GetSymbol() for atom in self.rdkit_mol.GetAtoms()]
        self.n_ats = len(self.symbols)
        self.calculated_atoms = self.n_ats
        bonds = []
        bond_types = {"SINGLE": 1,
                      "DOUBLE": 2,
                      "TRIPLE": 3,
                      "AROMATIC": 4}
        for bond in self.rdkit_mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            bond_type = bond_types[str(bond.GetBondType())]
            if a1 < a2:
                bonds.append((a1, a2, bond_type))
            else:
                bonds.append((a2, a1, bond_type))
        self.bonds = np.array(bonds, dtype=np.int32)
        self.num_of_bonds = len(self.bonds)
        coordinates = []
        for i in range(0, self.rdkit_mol.GetNumAtoms()):
            pos = self.rdkit_mol.GetConformer().GetAtomPosition(i)
            coordinates.append((pos.x, pos.y, pos.z))
        self.coordinates = np.array(coordinates, dtype=np.float32)
        ats_sreprba = self.create_ba()
        bonds_srepr = [f"{'-'.join(sorted([ats_sreprba[ba1], ats_sreprba[ba2]]))}-{bond_type}"
                       for ba1, ba2, bond_type in bonds]

        # control, whether molecule consist of standart aminoacids
        problematic_atoms = {}
        for i, (atba, rdkit_at) in enumerate(zip(ats_sreprba,
                                                 self.rdkit_mol.GetAtoms())):
            if atba not in real_ats_types:
                res_info = rdkit_at.GetPDBResidueInfo()
                label_comp_id = f"{res_info.GetResidueName()}"
                label_seq_id = int(res_info.GetResidueNumber())
                label_atom_id = f"{res_info.GetName().strip()}"
                id = f"{label_comp_id} {label_seq_id} {label_atom_id}"
                message = problematic_atom_info(rdkit_at, atba, self.rdkit_mol)
                problematic_atoms[id] = {
                    "key": {
                        "labelCompId": label_comp_id,
                        "labelSeqId": label_seq_id,
                        "labelAtomId": label_atom_id,
                    },
                    "message": message,
                }
        if problematic_atoms:
            raise ValueError(problematic_atoms)


        # convert to numba data structure
        self.ats_srepr = List(ats_sreprba)
        self.bonds_srepr = List(bonds_srepr)

    def create_ba(self) -> list:
        bonded_ats = [[] for _ in range(self.n_ats)]
        for bonded_at1, bonded_at2, _ in self.bonds:
            bonded_ats[bonded_at1].append(self.symbols[bonded_at2])
            bonded_ats[bonded_at2].append(self.symbols[bonded_at1])
        return [f"{symbol}/{''.join(sorted(bonded_ats))}"
                for symbol, bonded_ats in zip(self.symbols, bonded_ats)]

