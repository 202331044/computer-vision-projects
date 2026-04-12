import torch.optim as optim
from model import SimpleNN, Task1NN, Task2NN
import pickle
from sklearn.model_selection import StratifiedKFold
import numpy as np
import random
import torch

def get_model(name):

   if name == "baseline":
      return SimpleNN()
   elif name == "task1":
      return Task1NN()
   elif name == 'task2':
      return Task2NN()   
   else:
      raise ValueError("Unknow model")


def get_optimizer(name, model):
   
   if name == "SGD":
      return optim.SGD(model.parameters())
   elif name == "Adam":
      return optim.Adam(model.parameters())
   elif name == "AdamW":
      return optim.AdamW(model.parameter(), weight_decay=1e-4)
   elif name == "Momentum":
      return optim.SGD(model.parameters(), momentum=0.9)
   else:
      raise ValueError("Unknow optimizer")


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
