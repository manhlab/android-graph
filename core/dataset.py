from typing import List, Dict

import dgl
import torch
from torch.utils.data import Dataset

attributes = {"external", "entrypoint", "native", "public", "static", "codesize"}


class MalwareDataset(Dataset):
    def __init__(
        self,
        source_dir: List[str],
        labels: List[int],
    ):
        self.source_dir = source_dir
        self.labels = labels
        self.consider_features = ["user", "api"]

    def __len__(self) -> int:
        """Denotes the total number of samples"""
        return len(self.labels)

    def __getitem__(self, index: int):
        """Generates one sample of data"""
        try:
            graphs, _ = dgl.data.utils.load_graphs(str(self.source_dir[index]))
            graph: dgl.DGLGraph = dgl.add_self_loop(graphs[0])
            g = self._process_node_attributes(graph)
            if len(g.ndata.keys()) > 0:
                features = torch.cat(
                    [g.ndata[x] for x in self.consider_features], dim=1
                ).float()
            else:
                features = (g.in_degrees() + g.out_degrees()).view(-1, 1).float()
            g.ndata.clear()
            g.ndata["features"] = features
            return g, torch.tensor(self.labels[index])
        except:
            print(self.samples[index])

    @staticmethod
    def _process_node_attributes(g: dgl.DGLGraph):
        for attribute in attributes & set(g.ndata.keys()):
            g.ndata[attribute] = g.ndata[attribute].view(-1, 1)
        return g
