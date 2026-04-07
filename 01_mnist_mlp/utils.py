import torch.optim as optim
from model import SimpleNN, Task1NN

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