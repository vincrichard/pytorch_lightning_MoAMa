from typing import List, Protocol, Optional
import multiprocessing
from pathlib import Path
import pickle
import logging

import pandas as pd
import torch
from torch.utils.data import Dataset
from rdkit import Chem, DataStructs
from torch_geometric.data import Data, Batch
from jaxtyping import Float, Int, Bool
import numpy as np
from tqdm import tqdm

from src.featurizer import SimpleGraph2dFeaturizer, MotifMaskAtom

logger = logging.getLogger("MoAMa")


class MoAMaBatch(Protocol):
    x: Float[torch.Tensor, "num_atom atom_feat"]
    """Float[torch.Tensor, "num_atom atom_feat"]"""
    edge_index: Int[torch.Tensor, "2 num_edge"]
    """Int[torch.Tensor, "2 num_edge"]"""
    edge_attr: Float[torch.Tensor, "num_edge attr_feat"]
    """Float[torch.Tensor, "num_edge attr_feat"]"""
    batch: Float[torch.Tensor, " num_atom"]
    """Float[torch.Tensor, "num_atom"]"""
    smiles: List[str]
    fingerprint: List[DataStructs.cDataStructs.ExplicitBitVect]
    tanimoto_sim: Float[torch.Tensor, "B B"]
    """Float[torch.Tensor, "B B"]"""
    node_mask: Bool[torch.Tensor, " num_atom"]
    """Bool[torch.Tensor, "num_atom"]"""
    node_target: Float[torch.Tensor, "num_mask_atom num_atom_type"]
    """Float[torch.Tensor, "B atom_feat"]"""
    mask_node_label: Float[torch.Tensor, "num_mask_atom atom_feat"]
    """Float[torch.Tensor, "B num_atom_type"]"""


class Zinc(Dataset):

    def __init__(self, dataset_path: str, masking_strategy: MotifMaskAtom):
        """Dataset is expected to be be a smiles column without header."""
        super().__init__()
        self.smiles = pd.read_csv(dataset_path, header=None)[0]
        self.dataset_path = dataset_path
        self.featurizer = SimpleGraph2dFeaturizer()
        self.masking_strategy = masking_strategy
        self.preprocess_fingerprint = False

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx) -> Data:
        data = self.masking_strategy(
            self.featurizer(self.smiles[idx]), self.smiles[idx]
        )
        data.smiles = self.smiles[idx]
        if self.preprocess_fingerprint:
            data.fingerprint = DataStructs.CreateFromBitString(self.fingerprints[idx])
        else:
            data.fingerprint = Zinc.get_fingerprint(data.smiles)
        return data

    def pre_compute_fingerprints(self, path_fingerprint: Optional[str] = None):
        self.path_fingerprint = (
            path_fingerprint
            if path_fingerprint is not None
            else Path(self.dataset_path).parent / "fingerprints.pkl"
        )
        self.fingerprints = self.get_fingerprints(self.smiles)
        self.preprocess_fingerprint = True

    def get_fingerprints(
        self, smiles
    ) -> List[DataStructs.cDataStructs.ExplicitBitVect]:
        if Path(self.path_fingerprint).exists():
            return self.load_fingerprints()
        else:
            return self.compute_fingerprints(smiles)

    def load_fingerprints(self):
        logger.info(f"Loading fingerprints from {self.path_fingerprint}...")
        try:
            with open(self.path_fingerprint, "rb") as handle:
                return pickle.load(handle)
        except Exception as e:
            raise ChildProcessError(
                "There seems to be an issue with the fingerprint loading."
                f"Try to delete the file {self.path_fingerprint} and relaunch the training"
            ) from e

    def compute_fingerprints(
        self, smiles
    ) -> List[DataStructs.cDataStructs.ExplicitBitVect]:
        logger.info("Computing fingeprints...")
        with multiprocessing.Pool(20) as executor:
            fingerprints = [
                fp.ToBitString()
                for fp in tqdm(
                    executor.imap(Zinc.get_fingerprint, smiles, chunksize=100),
                    total=len(smiles),
                )
            ]

        logger.info(f"Saving fingeprints to {self.path_fingerprint}...")
        with open(self.path_fingerprint, "wb") as handle:
            pickle.dump(fingerprints, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return fingerprints

    @staticmethod
    def get_fingerprint(smiles):
        return Chem.RDKFingerprint(Chem.MolFromSmiles(smiles))


def similarity_collator(batch: List[Data]):
    batch = Batch.from_data_list(batch)
    batch.tanimoto_sim = compute_batch_tanimoto_similarity(batch)
    return batch


def compute_batch_tanimoto_similarity(batch: MoAMaBatch) -> Float[torch.Tensor, "B B"]:
    size_batch = len(batch.smiles)
    tanimoto_sim = np.zeros([size_batch, size_batch])
    for i in range(size_batch):
        tanimoto_sim[i, :i] = DataStructs.BulkTanimotoSimilarity(
            batch.fingerprint[i], batch.fingerprint[:i]
        )
    return torch.from_numpy(tanimoto_sim)
