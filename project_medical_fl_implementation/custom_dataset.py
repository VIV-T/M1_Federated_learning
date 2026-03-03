# Dans un fichier, par exemple my_dataset.py
from fluke.data.base import BaseDataset
from fluke.data import register_dataset

import pandas as pd
from typing import Tuple
import numpy as np

class MyCSVDataset(BaseDataset):
    def __init__(self, path: str, target: str, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.target = target
        self.data = pd.read_csv(path)
        self.X = self.data.drop(columns=[target]).values
        self.y = self.data[target].values

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        return self.X[idx], self.y[idx]

# Enregistrer le dataset pour fluke
register_dataset("my_csv", MyCSVDataset)
