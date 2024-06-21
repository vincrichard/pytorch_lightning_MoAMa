from typing import List, Dict, Union
import random
from pathlib import Path
import multiprocessing
import logging
import pickle

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np
from rdkit import Chem
from tqdm import tqdm

from .motif_extractor import AbstractMotifExtractor, Motif

logger = logging.getLogger("MoAMa")


class MotifMaskAtom:
    def __init__(
        self,
        num_atom_type: int,
        mask_rate: float,
        motif_extractor: AbstractMotifExtractor,
        max_motif_depth: int = 5,
    ):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param mask_rate: % of atoms/motifs to be masked
        :param max_motif_depth: maximum k hop necessary to retrieve the motif information.
        :param smiles_list: If the smiles list is provided we will compute the motif before
        training, to speed up the process.
        """
        self.num_atom_type = num_atom_type
        self.mask_rate = mask_rate
        self.preprocess_motif = False
        self.max_motif_depth = max_motif_depth
        self.motif_extractor = motif_extractor

    def __call__(self, data: Data, smiles: str, masked_atom_indices=None) -> Data:
        """
        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        """
        if self.preprocess_motif:
            motifs = self.motifs[smiles]
        else:
            _, motifs = self.compute_motifs_for_smiles(smiles)
        masked_atom_indices, unmaskable_atoms = self.get_motif_base_mask(data, motifs)
        masked_atom_indices = self.additional_random_atom_mask(data, masked_atom_indices, unmaskable_atoms)

        # Change id to mask to be able to use torch_geometric dataloader
        node_mask = torch.zeros(data.x.size(0), dtype=torch.bool)
        node_mask[masked_atom_indices] = 1
        data.node_mask = node_mask

        data.mask_node_label = data.x[masked_atom_indices]
        data.node_target = F.one_hot(data.mask_node_label[:, 0], num_classes=self.num_atom_type).float()

        # Mask with num_atom_type and 0 for no chirality
        data.x[masked_atom_indices] = torch.tensor([self.num_atom_type, 0])
        return data

    def pre_compute_motifs(self, smiles: List[str], dir_to_save: str):
        self.preprocess_motif = True
        path_motifs = Path(dir_to_save) / f"{self.motif_extractor.__class__.__name__}_motifs.pkl"
        self.motifs = self.get_motifs(smiles, path_motifs)

    def additional_random_atom_mask(
        self, data: Data, masked_atom_indices: List[int], unmaskable_atoms: List[int]
    ) -> List[int]:
        """In case the masking of random motif is not sufficient additional masking is applied.
        unmaskable_atoms are atoms neigboring large selected motif and so should not be masked.
        """
        num_atoms = data.x.size(0)
        sample_size = int(num_atoms * self.mask_rate + 1)
        if len(masked_atom_indices) < sample_size:
            remaining_atoms = set(range(num_atoms)) - set(masked_atom_indices) - set(unmaskable_atoms)
            random_masked_atoms = random.sample(remaining_atoms, sample_size - len(masked_atom_indices))
            masked_atom_indices.extend(random_masked_atoms)
        return masked_atom_indices

    def get_motif_base_mask(self, data: Data, motifs: List[Motif]) -> Union[List[int], List[int]]:
        """Randomly sample motif and adds it to the masked atoms. If the motif is already of size
        max_motif_depth, adds neighbors_id to unmaskable_atoms to avoid having a deeper motif.

        returns:
        masked_atom_indices: Atom to be masked
        unmaskable_atoms: Neigboring atoms to remove from additional random atom masking.
        """
        num_atoms = data.x.size(0)
        sample_size = int(num_atoms * self.mask_rate + 1)
        masked_atom_indices, unmaskable_atoms = [], []
        src, dst = data.edge_index

        while len(masked_atom_indices) < sample_size and len(motifs):
            candidate = motifs.pop(random.randint(0, len(motifs) - 1))
            if len(masked_atom_indices) + len(candidate.atom_ids) > sample_size + 0.1 * num_atoms:
                continue

            candidate_neighboring_atoms = set(dst[np.isin(src, candidate.atom_ids)].tolist())
            motifs = [
                motif for motif in motifs if not self.is_neighboring_motif(motif.atom_ids, candidate_neighboring_atoms)
            ]

            masked_atom_indices.extend(candidate.atom_ids)
            if candidate.size_largest_parth == self.max_motif_depth:
                unmaskable_atoms.extend(candidate.neighbors_id)

        return masked_atom_indices, unmaskable_atoms

    def compute_motifs_for_smiles(self, smiles: str) -> Union[str, List[Motif]]:
        mol = Chem.MolFromSmiles(smiles)
        return smiles, self.motif_extractor.get_motifs(mol)

    def get_motifs(self, smiles_list: List[str], path_motifs: Path) -> Dict[str, List[Motif]]:
        if Path(path_motifs).exists():
            return self.load_motifs(path_motifs)
        else:
            return self.compute_motifs(smiles_list, path_motifs)

    def load_motifs(self, path_motifs: Path):
        logger.info(f"Loading motifs from {path_motifs}...")
        try:
            with open(path_motifs, "rb") as handle:
                return pickle.load(handle)
        except Exception as e:
            raise ChildProcessError(
                "There seems to be an issue with the motifs loading."
                f"Try to delete the file {path_motifs} and relaunch the training"
            ) from e

    def compute_motifs(self, smiles_list: List[str], path_motifs: Path) -> Dict[str, List[Motif]]:
        logger.info("Computing motifs...")
        with multiprocessing.Pool(20) as executor:
            motifs = {
                smiles: motif
                for smiles, motif in tqdm(
                    executor.imap_unordered(self.compute_motifs_for_smiles, smiles_list),
                    total=len(smiles_list),
                )
            }

        logger.info(f"Saving motif to {path_motifs}...")
        with open(path_motifs, "wb") as handle:
            pickle.dump(motifs, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return motifs

    def is_neighboring_motif(self, motif, candidate_neighboring_atoms):
        return len(set(motif) & candidate_neighboring_atoms)

    def __repr__(self):
        return "{}(num_atom_type={}, mask_rate={})".format(
            self.__class__.__name__,
            self.num_atom_type,
            self.mask_rate,
        )
