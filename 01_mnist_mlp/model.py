import torch.nn as nn
import torch.nn.functional as F # functional operations

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