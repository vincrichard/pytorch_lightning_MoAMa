from typing import Union

import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch
import numpy as np

from src.featurizer import SimpleGraph2dFeaturizer


class Tox21(Dataset):

    def __init__(
        self,
        path,
        labels_columns=[
            "NR-AR",
            "NR-AR-LBD",
            "NR-AhR",
            "NR-Aromatase",
            "NR-ER",
            "NR-ER-LBD",
            "NR-PPAR-gamma",
            "SR-ARE",
            "SR-ATAD5",
            "SR-HSE",
            "SR-MMP",
            "SR-p53",
        ],
    ):
        super().__init__()
        self._dataset = pd.read_csv(path)
        self.smiles = self._dataset["smiles"]
        self.labels_columns = labels_columns
        self.num_target = len(self.labels_columns)
        self.featurizer = SimpleGraph2dFeaturizer()

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx) -> Union[Data, np.ndarray]:
        entry = self._dataset.loc[idx]
        graph_data = self.featurizer(entry["smiles"])
        graph_data.id = idx
        graph_data.y = torch.FloatTensor(entry[self.labels_columns]).view(1, -1)
        return graph_data
