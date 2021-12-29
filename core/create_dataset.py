from sklearn.model_selection import StratifiedKFold
import os
import pandas as pd
from os import listdir
import dgl
def gen_csv_file(dir_path):
    files = []
    file_name = []
    label = []
    num_nodes = []
    num_edges = []
    for  f in listdir(dir_path):
        for k in listdir(os.path.join(dir_path, f)):
                try:
                    graphs, labels = dgl.data.utils.load_graphs(str(os.path.join(dir_path, f, k)))
                    graph: dgl.DGLGraph = graphs[0]
                    num_nodes.append(graph.num_nodes())
                    num_edges.append(graph.num_edges())
                    files.append(str(os.path.join(dir_path, f, k)))
                    file_name.append(k)
                    label.append(f)
                except:
                    pass
                
    df = pd.DataFrame.from_dict(
        {"file": files, "file_name": file_name, "label": label}
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (_, val_) in enumerate(skf.split(X=df, y=df.label)):
        df.loc[val_, "fold"] = int(fold)
    df.to_csv("dataset.csv", index=False)

if __name__ == "__main__":
    gen_csv_file("preprecessing")