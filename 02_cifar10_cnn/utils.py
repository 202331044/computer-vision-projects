import torch
import torch.optim as optim
import numpy as np
from model import CNN as BaselineCNN
from model import Task1CNN
import pickle
from sklearn.model_selection import StratifiedKFold
import random

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

def split_train_val_data(datasets, train_ratio=0.9, seed=42):

    train_indices = []
    val_indices = []
    targets = np.array(datasets.targets)
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

    train_datasets = torch.utils.data.Subset(datasets, train_indices)
    val_datasets= torch.utils.data.Subset(datasets, val_indices)

    return train_datasets, val_datasets


def make_train_val_data(datasets, n_splits=5, save_path="splits.pkl"):
  
   skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

   splits = []
   for train_idx, val_idx in skf.split(datasets.data, datasets.targets):
      splits.append((train_idx, val_idx))

   with open(save_path, "wb") as f:
      pickle.dump(splits, f)

   return splits


def load_train_val_data(load_file="splits.pkl"):
   with open(load_file, "rb") as f:
      splits = pickle.load(f)
      
   return splits


def set_seed(seed=42):
   random.seed(seed)
   np.random.seed(seed)

   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)

   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark=False