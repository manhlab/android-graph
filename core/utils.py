import string
import torch
import random
import numpy as np
import os
from typing import List, Dict, Tuple
import dgl

LABELS = ["Adware", "Banking", "SMS", "Benign", "Riskware"]


def wandb_id_generator(size=12, chars=string.ascii_lowercase + string.digits):
    return "".join(random.SystemRandom().choice(chars) for _ in range(size))


def seed_everything(seed):
    """seed all random variables

    Args:
        seed (int): the random state
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_labels(path: List[str]):
    labels = []
    for apk in path:
        name = apk.split("/")[-2]
        labels.append(int(LABELS.index(name)))
    return labels


@torch.no_grad()
def collate_func(samples: List[Tuple[dgl.DGLGraph, int]]):
    """
    Batches several graphs into one
    :param samples: Tuple containing graph and its label
    :return: Batched graph, and labels concatenated into a tensor
    """
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels)
    return batched_graph, labels.float()
