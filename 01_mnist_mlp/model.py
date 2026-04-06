import torch.nn as nn
import torch.nn.functional as F # functional operations
import torch.optim as optim

def get_model(name):

    if name == "baseline":
        return SimpleNN()
    elif name == "task1":
        return Task1NN()
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


class SimpleNN(nn.Module):# All PyTorch models must inherit nn.Module
  
  def __init__(self):
    super().__init__()
    
    self.fc1 = nn.Linear(28*28, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    flat_x = x.view(-1, 28*28)

    x1 = F.relu(self.fc1(flat_x))
    x2 = self.fc2(x1)

    return x2


class Task1NN(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        flat_x = x.view(-1, 28*28)

        x1 = F.relu(self.fc1(flat_x))
        x2 = self.fc2(x1)

        return x2