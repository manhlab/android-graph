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

LABELS = [ "Adware",  "Banking", "SMS", "Benign", "Riskware"]
def stratified_split_dataset(
    samples: List[str], labels: Dict[str, int], ratios: Tuple[float, float]
) -> Tuple[List[str], List[str]]:
    """
    Split the dataset into train and validation datasets based on the given ratio
    :param samples: List of file names
    :param labels: Mapping from file name to label
    :param ratios: Training ratio, validation ratio
    :return: List of file names in training and validation split
    """
    if sum(ratios) != 1:
        raise Exception("Invalid ratios provided")
    train_ratio, val_ratio = ratios
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=0)
    train_idx, val_idx = list(sss.split(samples, [labels[x] for x in samples]))[0]
    train_list = [samples[x] for x in train_idx]
    val_list = [samples[x] for x in val_idx]
    return train_list, val_list


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
        train_dir: Union[str, Path],
        test_dir: Union[str, Path],
        batch_size: int,
        split_ratios: Tuple[float, float],
        num_workers: int,
        pin_memory: bool,
        split_train_val: bool,
    ):
        super().__init__()
        self.train_dir = Path(train_dir)
        self.test_dir = Path(test_dir)
        self.dataloader_kwargs = {
            "num_workers": num_workers,
            "batch_size": batch_size,
            "pin_memory": pin_memory,
            "collate_fn": collate,
            "drop_last": True,
        }
        self.split_ratios = split_ratios
        self.split = split_train_val
        self.splitter = stratified_split_dataset

    @staticmethod
    def get_samples(path: List[str]) -> Tuple[List[str], Dict[str, int]]:
        samples = []
        labels = {}
        for apk in path:
            name = apk.split('/')[-1]
            samples.append(name)
            labels[name] = int(LABELS.index(apk.split('/')[1]))
        return samples, labels

    def setup(self, stage=None):
        df = pd.read_csv("dataset.csv")
        train_list = df[df['fold']!= 0].file.tolist()
        test_list = df[df['fold']== 0].file.tolist()
        samples, labels = self.get_samples(train_list)
        test_samples, test_labels = self.get_samples(test_list)
#         if self.split:
#             train_samples, val_samples = self.splitter(
#                 samples, labels, self.split_ratios
#             )
#             val_dir = self.train_dir
#             val_labels = labels
#         else:
#             train_samples = samples
#             val_dir = self.test_dir
#             val_samples, val_labels = test_samples, test_labels
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
