import os
from pathlib import Path
from typing import List, Dict, Tuple, Union

import dgl
import pytorch_lightning as pl
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
import pandas as pd
from core.dataset import MalwareDataset

LABELS = ["Adware", "Banking", "SMS", "Benign", "Riskware"]


@torch.no_grad()
def collate(samples: List[Tuple[dgl.DGLGraph, int]]):
    """
    Batches several graphs into one
    :param samples: Tuple containing graph and its label
    :return: Batched graph, and labels concatenated into a tensor
    """
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels)
    return batched_graph, labels.float()


class MalwareDataModule(pl.LightningDataModule):
    """
    Handler class for data loading, splitting and initializing datasets and dataloaders.
    """

    def __init__(
        self,
        data_src: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ):
        super().__init__()
        self.data_src = Path(data_src)
        self.dataloader_kwargs = {
            "num_workers": num_workers,
            "batch_size": batch_size,
            "pin_memory": pin_memory,
            "collate_fn": collate,
            "drop_last": True,
        }

    @staticmethod
    def get_samples(path: List[str]) -> Tuple[List[str], Dict[str, int]]:
        samples = []
        labels = {}
        for apk in path:
            name = apk.split("/")[-1]
            samples.append(name)
            labels[name] = int(LABELS.index(apk.split("/")[1]))
        return samples, labels

    def setup(self, stage=None):
        df = pd.read_csv(self.data_src)
        train_list = df[df["fold"] != 0].file.tolist()
        test_list = df[df["fold"] == 0].file.tolist()
        samples, labels = self.get_samples(train_list)
        test_samples, test_labels = self.get_samples(test_list)
        self.train_dataset = MalwareDataset(
            source_dir=train_list,
            samples=samples,
            labels=labels,
        )
        self.val_dataset = MalwareDataset(
            source_dir=test_list,
            samples=test_samples,
            labels=test_labels,
        )
        self.test_dataset = MalwareDataset(
            source_dir=test_list,
            samples=test_samples,
            labels=test_labels,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.dataloader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.dataloader_kwargs)
