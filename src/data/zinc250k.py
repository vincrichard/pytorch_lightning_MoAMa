from functools import partial

import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from src.featurizer import Graph2dFeaturizer


class Zinc250(Dataset):
    BASE_PATH = "./data/"
    ZINC_PATH = f"{BASE_PATH}zinc250k.csv"

    def __init__(self):
        super().__init__()
        self.smiles = pd.read_csv(self.ZINC_PATH)["smiles"].tolist()
        self.featurizer = Graph2dFeaturizer()

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.featurizer(self.smiles[idx])


Zinc250_DataLoader = partial(DataLoader, Zinc250())
