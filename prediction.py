import warnings
import sklearn.exceptions
from sklearn import metrics

# General
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import gc
import os
from pathlib import Path
# Deep Learning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from core import *
import dgl
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
from core.process_dataset import process as create_graph
gc.enable()
seed_everything(CFG.RANDOM_SEED)
CFG.HASH_NAME = wandb_id_generator(size=12)
# Device Optimization
CFG.device = "cuda" if torch.cuda.is_available() else "cpu"
CFG.consider_features = ["user", "entrypoint", "api"]

ATTRIBUTES = [
    "external",
    "entrypoint",
    "native",
    "public",
    "static",
    "codesize",
    "api",
    "user",
]
attributes = {"external", "entrypoint", "native", "public", "static", "codesize"}

LABELS = ["Adware", "Banking", "SMS", "Benign", "Riskware"]

def _process_node_attributes(g: dgl.DGLGraph):
    for attribute in attributes & set(g.ndata.keys()):
        g.ndata[attribute] = g.ndata[attribute].view(-1, 1)
    return g
def inference(path, CFG):
    print(f"========== Start training ==========")
    dest_path = Path("apk")
    create_graph(path, dest_path)
    file_name = path.stem
    dest_dir = dest_path / f"{file_name}.fcg"
    try:
        graphs, _ = dgl.data.utils.load_graphs(str(dest_dir))
        graph: dgl.DGLGraph = dgl.add_self_loop(graphs[0])
        g = _process_node_attributes(graph)
        if len(g.ndata.keys()) > 0:
            features = torch.cat(
                [g.ndata[x] for x in CFG.consider_features], dim=1
            ).float()
        else:
            features = (g.in_degrees() + g.out_degrees()).view(-1, 1).float()
        g.ndata.clear()
        g.ndata["features"] = features
    except:
        print("File not valid")

    model = MalwareDetector(
        input_dimension=CFG.input_dimension,
        convolution_algorithm=CFG.convolution_algorithm,
        convolution_count=CFG.convolution_count,
    )
    model.to(CFG.device)
    model.load_state_dict(torch.load("GraphConv_best.pth" , map_location=torch.device('cpu')), strict=True)
    g = dgl.batch([g])
    # g = g.to(CFG.device)
    output = model(g)
    output = torch.softmax(output, dim=0)
    
    torch.cuda.empty_cache()
    gc.collect()
    return output


if __name__ == "__main__":
    print(f"Using device: {CFG.device}")
    score = inference(Path("apk/010174a982ff9262e95d779779d7be32fddc57f08d17dbd49e0bc31bbbc4e6bc.d0ad3be96dbcc89af12436bc9e6944fd"), CFG)
    
    label= np.argmax(score.detach().cpu().numpy(), axis=0)
    label = LABELS[label]
    print(score, label)