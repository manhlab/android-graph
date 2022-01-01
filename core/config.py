import os


class CFG:
    # configs for training
    RANDOM_SEED = 42
    data_dim = 49  # dimension of a vector
    num_classes = 2  # 2 classes
    batch_size = 256
    input_dimension = 128  # number of units by layer
    convolution_count = 3
    convolution_algorithm = "GraphConv"
    epochs = 15
    num_workers = 2
    batch_size = 8  # number of batch_size
    T_max = 15
    lr = 0.005
    gradient_accumulation_steps = 1
    WANDB_KEY = "caf7c9698a3a861cd4b921ed5a14f9b0105d7cbe"
    encoders_path = "./encoders"
    extracted_path = "./preprocessing"
    data_path = "/home/manhlab/dataset/Dataset"
    model_path = "/home/manhlab/model"
    temp_dir = "./transferedFiles"
    result_dir = "./analysis"
    file_name = "./exe_file/rufus-3.13p.exe"
    device = "cpu"
