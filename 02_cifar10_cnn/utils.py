import torch
import torch.optim as optim
import numpy as np
from model import CNN as BaselineCNN
from model import Task1CNN

def get_model(model_name):
    if model_name == "baseline":
        return BaselineCNN()
    elif model_name == "task1":
        return Task1CNN()
    else:
        raise ValueError("Unkown model")

def get_optimizer(opt_name, model):
    if opt_name == "SGD":
        return optim.SGD(model.parameters(), weight_decay=1e-4)
    elif opt_name == "Adam":
        return optim.Adam(model.parameters(), lr=0.001)
    elif opt_name == "AdamW":
        return optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    elif opt_name == "Momentum":
        return optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    else:
        raise ValueError("Unkown optimizer")

def split_train_val_data(full_train_datasets, train_ratio = 0.9, seed = 42):
    train_indices = []
    val_indices = []
    targets = np.array(full_train_datasets.targets)
    class_numbers = np.unique(targets)
    np.random.seed(seed)

    for i in class_numbers:
      class_indices = np.where((i == targets))[0]
      np.random.shuffle(class_indices)

      train_size = int(train_ratio * len(class_indices))
      train_indices.extend(class_indices[:train_size])
      val_indices.extend(class_indices[train_size:])
    
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    train_datasets = torch.utils.data.Subset(full_train_datasets, train_indices)
    val_datasets= torch.utils.data.Subset(full_train_datasets, val_indices)

    return train_datasets, val_datasets
