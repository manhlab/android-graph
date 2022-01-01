import os


class CFG:
    # configs for training
    RANDOM_SEED = 42
    input_dimension = 247  # number of units by layer
    convolution_count = 3
    convolution_algorithm = "GraphConv"
    epochs = 15
    num_workers = 20
    batch_size = 8  # number of batch_size
    T_max = 15
    lr = 0.005
    gradient_accumulation_steps = 1
    WANDB_KEY = "caf7c9698a3a861cd4b921ed5a14f9b0105d7cbe"
    model_path = "model"
    device = "cpu"
