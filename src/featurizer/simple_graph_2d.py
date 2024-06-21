from typing import List, Tuple

import torch
from torch_geometric.data import Data as TorchGeometricData
import numpy as np
from rdkit import Chem, RDLogger

from src.featurizer.dgllife import (
    atom_chiral_tag_one_hot,
    bond_type_one_hot,
    bond_direction_one_hot,
)


class SimpleGraph2dFeaturizer:
    """
    Atom features: Atom type, Chirality type
    Edge features: Bond type, Bond direction
    """

    LIST_BOND_FEATURIZERS = [
        lambda bond: [bond_type_one_hot(bond).index(True)],
        lambda bond: [bond_direction_one_hot(bond).index(True)],
    ]

    ALLOW_ATOM_TYPES = list(range(1, 119))

    CHIRAL_TYPES = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ]

    BOND_TYPES = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]

    BOND_DIR = [
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
    ]

    def _get_vertex_features(self, mol: Chem.Mol) -> List[List[float]]:
        return np.array([self._featurize_atom(atom) for atom in mol.GetAtoms()])

    def _featurize_atom(self, atom: Chem.Atom) -> List[float]:
        return np.concatenate([featurizer(atom) for featurizer in self._list_atom_featurizers()])

    def _list_atom_featurizers(self):
        return [
            # lambda atom: [atom_type_one_hot(atom, allowable_set=self.allowed_atom_types, encode_unknown=False).index(True)],
            lambda atom: [atom.GetAtomicNum() - 1],
            lambda atom: [atom_chiral_tag_one_hot(atom).index(True)],
        ]

    def _get_edge_features(self, mol: Chem.Mol) -> Tuple[List[List[int]], List[List[float]]]:
        edge_indices, edge_attributes = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_features = self._featurize_bond(bond)

            edge_indices.extend([[i, j], [j, i]])
            edge_attributes.extend([bond_features] * 2)
        if not len(edge_indices):
            return np.empty(shape=(0, 2)), np.empty(shape=(0, 2))

        return np.array(edge_indices), np.array(edge_attributes)

    def _featurize_bond(self, bond: Chem.Bond) -> List[float]:
        return np.concatenate([featurizer(bond) for featurizer in self.LIST_BOND_FEATURIZERS])

    def __call__(self, smiles: str) -> TorchGeometricData:
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        RDLogger.EnableLog("rdApp.*")

        atom_features = self._get_vertex_features(mol)
        atom_features = torch.LongTensor(atom_features).view(-1, len(atom_features[0]))

        edge_indices, edge_attributes = self._get_edge_features(mol)
        edge_indices = torch.tensor(edge_indices, dtype=torch.long).t()
        edge_attributes = torch.LongTensor(edge_attributes)

        return TorchGeometricData(
            x=atom_features,
            edge_index=edge_indices,
            edge_attr=edge_attributes,
            smiles=smiles,
        )

    def decode(self, data: TorchGeometricData) -> Chem.Mol:
        mol = Chem.RWMol()

        # atoms
        atom_features = data.x.cpu().numpy()
        num_atoms = atom_features.shape[0]
        for i in range(num_atoms):
            atomic_num_idx, chirality_tag_idx = atom_features[i]
            atomic_num = self.ALLOW_ATOM_TYPES[int(atomic_num_idx)]
            chirality_tag = self.CHIRAL_TYPES[int(chirality_tag_idx)]
            atom = Chem.Atom(atomic_num)
            atom.SetChiralTag(chirality_tag)
            mol.AddAtom(atom)

        # bonds
        edge_index = data.edge_index.cpu().numpy()
        edge_attr = data.edge_attr.cpu().numpy()
        num_bonds = edge_index.shape[1]
        for j in range(0, num_bonds, 2):
            begin_idx = int(edge_index[0, j])
            end_idx = int(edge_index[1, j])
            bond_type_idx, bond_dir_idx = edge_attr[j]
            bond_type = self.BOND_TYPES[int(bond_type_idx)]
            bond_dir = self.BOND_DIR[int(bond_dir_idx)]
            mol.AddBond(begin_idx, end_idx, bond_type)
            # set bond direction
            new_bond = mol.GetBondBetweenAtoms(begin_idx, end_idx)
            new_bond.SetBondDir(bond_dir)

        return mol
