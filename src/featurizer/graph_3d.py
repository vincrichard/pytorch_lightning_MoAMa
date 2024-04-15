from typing import List, Tuple
import itertools
from functools import partial
from pathlib import Path

from rdkit import Chem
import torch
from torch_geometric.data import Data as TorchGeometricData
from torch_geometric.utils import k_hop_subgraph
import numpy as np

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


class Graph3dFeaturizer:
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
        folder_data_path: str = "",
        allowed_atom_types: List[str] = None,
    ):
        if allowed_atom_types is None:
            allowed_atom_types = self.DEFAULT_ATOM_TYPES
        self.allowed_atom_types = allowed_atom_types
        self.folder_data_path = Path(folder_data_path)

    def _read_data(self, file_path: str, **kwargs):
        if file_path.endswith(".mol2"):
            return Chem.MolFromMol2File(file_path, **kwargs)

        if file_path.endswith(".sdf"):
            return next(Chem.SDMolSupplier(file_path, **kwargs))

        if file_path.endswith(".pdb"):
            return Chem.MolFromPDBFile(file_path, **kwargs)

        raise NotImplementedError

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

    def _get_3d_edge_features(self, coords, edge_indices):
        # This part is also taken from IGN
        edge_attributes = []
        for src_node, dst_node in zip(*edge_indices.tolist()):
            neighboring_nodes, _, _, _ = k_hop_subgraph(src_node, 1, edge_indices)
            neighboring_nodes = neighboring_nodes[neighboring_nodes != src_node]
            neighboring_nodes = neighboring_nodes[neighboring_nodes != dst_node]
            edge_attributes.append(
                compute_3d_edge_attr(src_node, dst_node, neighboring_nodes, coords)
            )
        return torch.FloatTensor(edge_attributes)

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

    def __call__(self, path: str) -> TorchGeometricData:
        mol = self._read_data(str(self.folder_data_path / path))
        if mol is None:
            raise ValueError(
                f"Issue while reading the file during 3D featurization {path}"
            )

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

            # Generates IGN 10 features
            edge_attributes_3d = self._get_3d_edge_features(
                coords=mol.GetConformer().GetPositions(), edge_indices=edge_indices
            )
            edge_attributes = torch.concat([edge_attributes, edge_attributes_3d], dim=1)

        return TorchGeometricData(
            x=atom_features,
            edge_index=edge_indices,
            edge_attr=edge_attributes,
            smiles=Chem.MolToSmiles(mol),
        )


# Code taken (and refactored) form IGN to get 3D edge features


def get_distance_vector(a, b):
    return np.sqrt(np.sum((b - a) ** 2))


def triangle_area(angle_a, ab, ac):
    return 0.5 * ab * ac * np.sin(angle_a)


def get_angle(a, b, c):
    """The angle in a is computed"""
    ab = b - a
    ac = c - a
    if (np.linalg.norm(ab) * np.linalg.norm(ac)) == 0:
        cosine_angle = -1
    else:
        cosine_angle = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))
    cosine_angle = cosine_angle if cosine_angle >= -1.0 else -1.0
    return np.arccos(cosine_angle)


def get_3d_statistic(a, b, c):
    angle = get_angle(a, b, c)
    ab = get_distance_vector(a, b)
    ac = get_distance_vector(a, c)
    area = triangle_area(angle, ab, ac)
    return np.degrees(angle), area, ac


def compute_3d_edge_attr(src_nodes, dst_node, neighboring_nodes, coords):
    """
    Return 10 values
    3 Angles statistics: max, sum, mean of scaled (multiplied by 0.001) angle between atoms i, j, k in 3D space (angle on i)
    3 Area statistics: max, sum, mean of areas between atoms i, j, k in 3D space
    3 Distance statistics: max, sum, mean of scaled (multiplied by 0.1) distances between atoms i, k in 3D space
    1 Distance: The scaled Euclidean distance (multiplied by 0.1) between the connected atoms in 3D space
    Taken from https://github.com/zjujdj/IGN/blob/master/scripts/graph_constructor.py and refactor.
    """
    euclidean = get_distance_vector(coords[src_nodes], coords[dst_node]) * 0.1
    if len(neighboring_nodes) > 0:
        Angles = []
        Areas = []
        Distances = []
        for node_id in neighboring_nodes:
            angle, area, distance = get_3d_statistic(
                coords[src_nodes], coords[dst_node], coords[node_id]
            )
            Angles.append(angle)
            Areas.append(area)
            Distances.append(distance)
        return [
            np.max(Angles) * 0.01,
            np.sum(Angles) * 0.01,
            np.mean(Angles) * 0.01,
            np.max(Areas),
            np.sum(Areas),
            np.mean(Areas),
            np.max(Distances) * 0.1,
            np.sum(Distances) * 0.1,
            np.mean(Distances) * 0.1,
            euclidean,
        ]
    else:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, euclidean]
