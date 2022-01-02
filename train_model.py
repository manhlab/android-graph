import warnings
import sklearn.exceptions
import sklearn

# General
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import gc
import os

# Deep Learning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from core import *

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

gc.enable()
seed_everything(CFG.RANDOM_SEED)
CFG.HASH_NAME = wandb_id_generator(size=12)
# Device Optimization
if torch.cuda.is_available():
    CFG.device = str(torch.device("cuda"))
else:
    CFG.device = str(torch.device("cpu"))


def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler):
    model.train()
    losses = AverageMeter()
    bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (malware, labels) in bar:
        malware = malware.to(CFG.device)
        labels = labels.to(CFG.device)
        batch_size = labels.size(0)
        output = model(malware)
        loss = criterion(output, labels.long())
        losses.update(loss.item(), batch_size)
        loss.backward()
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        bar.set_postfix(
            Epoch=epoch, Train_Loss=losses.avg, LR=optimizer.param_groups[0]["lr"]
        )
    return losses.avg


@torch.no_grad()
def val_fn(val_loader, model, criterion, epoch):
    model.eval()
    losses = AverageMeter()
    bar = tqdm(enumerate(val_loader), total=len(val_loader))
    labels_list = []
    prediction_list = []
    for _, (malware, labels) in bar:
        malware = malware.to(CFG.device)
        labels = labels.to(CFG.device)
        batch_size = labels.size(0)
        output = model(malware)
        loss = criterion(output, labels.long())
        losses.update(loss.item(), batch_size)

        bar.set_postfix(Epoch=epoch, Val_Loss=losses.avg)
        labels_list.append(labels.detach().cpu().numpy())
        prediction_list.append(np.argmax(output.detach().cpu().numpy(), axis=1))

    labels_list = np.hstack(labels_list)
    prediction_list = np.hstack(prediction_list)
    accuracy = sklearn.metrics.accuracy_score(labels_list, prediction_list)
    f1 = sklearn.metrics.f1_score(labels_list, prediction_list, average="macro")
    print(f"============Valid Accuracy: {accuracy}=========")
    print(f"============Valid F1: {f1}=========")
    return losses.avg


def loop(df, CFG):
    run = wandb.init(
        project="PRMalware",
        job_type="Train",
        tags=["lstm", f"{CFG.HASH_NAME}", "crossentropyloss"],
        name=f"{CFG.HASH_NAME}",
        anonymous="must",
    )

    print(f"========== Start training ==========")

    trn_idx = df[df["fold"] != 0].index
    val_idx = df[df["fold"] == 0].index

    train_folds = df.loc[trn_idx].reset_index(drop=True)
    valid_folds = df.loc[val_idx].reset_index(drop=True)
    train_labels = get_labels(train_folds.file.tolist())
    test_labels = get_labels(valid_folds.file.tolist())
    train_dataset = MalwareDataset(
        source_dir=train_folds.file.tolist(), labels=train_labels
    )
    val_dataset = MalwareDataset(
        source_dir=valid_folds.file.tolist(), labels=test_labels
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_func,
    )
    valid_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_func,
    )

    model = MalwareDetector(
        input_dimension=CFG.input_dimension,
        convolution_algorithm=CFG.convolution_algorithm,
        convolution_count=CFG.convolution_count,
    )
    model.to(CFG.device)
    wandb.watch(model, log_freq=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr, eps=1e-7)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.T_max, eta_min=1e-4
    )

    criterion = nn.CrossEntropyLoss()
    best_score = 1
    for epoch in range(CFG.epochs):
        # train
        train_epoch_loss = train_fn(
            train_loader, model, criterion, optimizer, epoch, scheduler
        )
        val_epoch_loss = val_fn(valid_loader, model, criterion, epoch)
        # Log the metrics
        wandb.log({"Train Loss": train_epoch_loss})
        wandb.log({"Valid Loss": val_epoch_loss})

        if val_epoch_loss < best_score:
            best_score = val_epoch_loss
            run.summary["Best Loss"] = best_score
            print(f"Epoch {epoch+1} - Save Best Score: {val_epoch_loss:.4f} Model")
            torch.save(
                model.state_dict(),
                os.path.join(CFG.model_path, f"/{CFG.convolution_algorithm}_best.pth"),
            )
    run.finish()
    torch.cuda.empty_cache()
    gc.collect()
    del model, optimizer, scheduler
    return best_score


if __name__ == "__main__":
    print(f"Using device: {CFG.device}")
    try:
        api_key = CFG.WANDB_KEY
        wandb.login(key=api_key)
        anony = None
    except:
        anony = "must"
    df = pd.read_csv("dataset.csv")
    loop(df, CFG)
