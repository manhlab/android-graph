from typing import Tuple, Optional

import dgl
import dgl.nn.pytorch as graph_nn
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from dgl.nn import Sequential
from torch import nn


class MalwareDetector(pl.LightningModule):
    def __init__(
        self,
        input_dimension: int,
        convolution_algorithm: str,
        convolution_count: int,
    ):
        super().__init__()
        # supported_algorithms = ["GraphConv", "SAGEConv", "TAGConv", "DotGatConv"]
        convolution_algorithm = "GraphConv"
        self.save_hyperparameters()
        self.convolution_layers = []
        convolution_dimensions = [64, 32, 16]
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
        self.classify = nn.Linear(16, 5)
        self.loss_func = nn.CrossEntropyLoss()

    @staticmethod
    def _get_convolution_layer(
        name: str, input_dimension: int, output_dimension: int
    ) -> Optional[nn.Module]:
        return {
            "GraphConv": graph_nn.GraphConv(
                input_dimension, output_dimension, activation=F.relu
            ),
            "SAGEConv": graph_nn.SAGEConv(
                input_dimension,
                output_dimension,
                activation=F.relu,
                aggregator_type="mean",
                norm=F.normalize,
            ),
            "DotGatConv": graph_nn.DotGatConv(
                input_dimension, output_dimension, num_heads=1
            ),
            "TAGConv": graph_nn.TAGConv(input_dimension, output_dimension, k=4),
        }.get(name, None)

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        with g.local_scope():
            h = g.ndata["features"]
            h = self.convolution_layers(g, h)
            g.ndata["h"] = h if len(self.convolution_layers) > 0 else h[0]
            hg = dgl.mean_nodes(g, "h")
            return self.classify(hg).squeeze()

    def training_step(self, batch: Tuple[dgl.DGLGraph, torch.Tensor], batch_idx: int):
        bg, label = batch
        logits = self.forward(bg)
        loss = self.loss_func(logits, label.long())
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple[dgl.DGLGraph, torch.Tensor], batch_idx: int):
        bg, label = batch
        logits = self.forward(bg)
        loss = self.loss_func(logits, label.long())
        # prediction = torch.softmax(logits, 1)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch: Tuple[dgl.DGLGraph, torch.Tensor], batch_idx: int):
        bg, label = batch
        logits = self.forward(bg)
        loss = self.loss_func(logits, label)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
        return optimizer  # ], [scheduler]
