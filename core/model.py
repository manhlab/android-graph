from typing import Tuple, Optional

import dgl
import dgl.nn.pytorch as graph_nn
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from dgl.nn import Sequential
from torch import nn


class MalwareDetector(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        convolution_algorithm: str,
        convolution_count: int,
    ):
        super().__init__()
        # supported_algorithms = ["GraphConv", "SAGEConv", "TAGConv", "DotGatConv"]
        self.convolution_layers = []
        convolution_dimensions = [128, 64, 32, 16]
        for dimension in convolution_dimensions[:convolution_count]:
            self.convolution_layers.append(
                self._get_convolution_layer(
                    name=convolution_algorithm,
                    input_dimension=input_dimension,
                    output_dimension=dimension,
                )
            )
            input_dimension = dimension
        self.convolution_layers = Sequential(*self.convolution_layers)
        self.last_dimension = input_dimension
        self.classify = nn.Linear(32, 5)
        self.loss_func = nn.CrossEntropyLoss()

    @staticmethod
    def _get_convolution_layer(
        name: str, input_dimension: int, output_dimension: int
    ) -> Optional[nn.Module]:
        return {
            "GraphConv": graph_nn.GraphConv(
                input_dimension, output_dimension, activation=F.leaky_relu, 
                norm="both", weight=True, bias=False
            ),
            "SAGEConv": graph_nn.SAGEConv(
                input_dimension,
                output_dimension,
                activation=F.relu,
                aggregator_type="mean",
                norm=F.normalize,
            ),
            "DotGatConv": graph_nn.DotGatConv(
                input_dimension, output_dimension, 3, 16
            ),
            "TAGConv": graph_nn.TAGConv(input_dimension, output_dimension, k=4),
            "GATConv": graph_nn.GATConv(input_dimension, output_dimension, num_heads=4),
        }.get(name, None)

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        with g.local_scope():
            h = g.ndata["features"]
            h = self.convolution_layers(g, h)
            g.ndata["h"] = h if len(self.convolution_layers) > 0 else h[0]
            hg = dgl.mean_nodes(g, "h")
            return self.classify(hg).squeeze()
