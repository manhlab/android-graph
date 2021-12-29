import os
from pathlib import Path
import yaml
import wandb
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from core import *

def train_model(cfg: DictConfig) -> None:
    data_module = MalwareDataModule(**cfg["data"])
    model = MalwareDetector(**cfg["model"])
    callbacks = [
        ModelCheckpoint(
            dirpath=os.getcwd(),
            filename=str("{epoch:02d}-{val_loss:.2f}.pt"),
            monitor="val_loss",
            mode="min",
            save_last=True,
            save_top_k=-1,
        )
    ]

    trainer_kwargs = dict(cfg["trainer"])
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule=data_module)
   
    # print(f"Using checkpoint {ckpt_path} for testing.")
    # model = MalwareDetector.load_from_checkpoint(ckpt_path, **cfg["model"])
    # trainer.test(model, datamodule=data_module, verbose=True)
    wandb.finish()


if __name__ == "__main__":
    with open("android-graph/config/conf.yaml", 'r') as f:
        config = yaml.safe_load(f) 
    train_model(config)
