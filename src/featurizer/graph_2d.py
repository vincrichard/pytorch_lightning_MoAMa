from typing import List, Tuple
import itertools
from functools import partial


from rdkit import Chem
import torch
from torch_geometric.data import Data as TorchGeometricData

from .dgllife import (
    atom_type_one_hot,
    atom_degree_one_hot,
    atom_implicit_valence_one_hot,
    atom_formal_charge_one_hot,
    atom_num_radical_electrons_one_hot,
    atom_hybridization_one_hot,
    atom_is_aromatic_one_hot,
    atom_total_num_H_one_hot,
    bond_type_one_hot,
    bond_is_conjugated,
    bond_is_in_ring,
    bond_stereo_one_hot,
)


class Graph2dFeaturizer:
    DEFAULT_ATOM_TYPES = [
        "B",
        "C",
        "N",
        "O",
        "F",
        "Na",
        "Si",
        "P",
        "S",
        "Cl",
        "K",
        "Br",
        "I",
    ]

    LIST_BOND_FEATURIZERS = [
        bond_type_one_hot,
        bond_is_conjugated,
        bond_is_in_ring,
        bond_stereo_one_hot,
    ]

    def __init__(
        self,
        allowed_atom_types: List[str] = None,
    ):
        if allowed_atom_types is None:
            allowed_atom_types = self.DEFAULT_ATOM_TYPES
        self.allowed_atom_types = allowed_atom_types

    def _get_vertex_features(self, mol: Chem.Mol) -> List[List[float]]:
        return [self._featurize_atom(atom) for atom in mol.GetAtoms()]

    def _featurize_atom(self, atom: Chem.Atom) -> List[float]:
        return list(
            itertools.chain.from_iterable(
                [featurizer(atom) for featurizer in self._list_atom_featurizers()]
            )
        )

    def _list_atom_featurizers(self):
        return [
            partial(
                atom_type_one_hot,
                allowable_set=self.allowed_atom_types,
                encode_unknown=True,
            ),
            atom_degree_one_hot,
            atom_implicit_valence_one_hot,
            atom_formal_charge_one_hot,
            atom_num_radical_electrons_one_hot,
            atom_hybridization_one_hot,
            atom_is_aromatic_one_hot,
            atom_total_num_H_one_hot,
        ]

    def _get_edge_features(
        self, mol: Chem.Mol
    ) -> Tuple[List[List[int]], List[List[float]]]:
        edge_indices, edge_attributes = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_features = self._featurize_bond(bond)

            edge_indices.extend([[i, j], [j, i]])
            edge_attributes.extend([bond_features] * 2)

        return edge_indices, edge_attributes

    def _featurize_bond(self, bond: Chem.Bond) -> List[float]:
        return list(
            itertools.chain.from_iterable(
                [featurizer(bond) for featurizer in self.LIST_BOND_FEATURIZERS]
            )
        )

    def sort_edges(self, number_of_atoms, edge_indices, edge_attributes):
        edge_unique_id = (edge_indices[0] * number_of_atoms + edge_indices[1]).argsort()
        permutation = edge_unique_id.argsort()
        return edge_indices[:, permutation], edge_attributes[permutation]

    def __call__(self, smiles: str) -> TorchGeometricData:
        mol = Chem.MolFromSmiles(smiles)

        # Compute all atom features a 54 vector per mol
        atom_features = self._get_vertex_features(mol)
        atom_features = torch.FloatTensor(atom_features).view(-1, len(atom_features[0]))

        # Compute all edge attributes a 22 vector per mol
        edge_indices, edge_attributes = self._get_edge_features(mol)
        edge_indices = torch.tensor(edge_indices, dtype=torch.long).t()
        edge_attributes = torch.FloatTensor(edge_attributes)

        if edge_indices.numel() > 0:
            edge_indices, edge_attributes = self.sort_edges(
                atom_features.size(0), edge_indices, edge_attributes
            )

        return TorchGeometricData(
            x=atom_features,
            edge_index=edge_indices,
            edge_attr=edge_attributes,
            smiles=smiles,
        )
