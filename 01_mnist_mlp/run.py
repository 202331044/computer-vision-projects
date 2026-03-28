import torch                                  # pytorch
import torchvision                            # image datasets + preprocessing tools
import torchvision.transforms as transforms   # preprocessing tools
import torch.nn as nn                         # layer definitions
import argparse

from model import SimpleNN as BaselineNN
from mini_tasks.task1 import SimpleNN as Task1NN 
from train import train, test

def get_model(name):
    if name == "baseline":
        return BaselineNN()
    elif name == "task1":
        return Task1NN()
    
def run(epochs, batch_size, name):
  
    # cuda = NVIDIA GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DL models accept only tensors, so convert image data (0~255 -> 0.0~1.0)
    transform = transforms.ToTensor()

    train_datasets = torchvision.datasets.MNIST (
        root = "./data", # location to store dataset
        train = True,
        transform = transform,
        download = True
    )

    test_datasets = torchvision.datasets.MNIST(
        root = "./data",
        train = False,
        transform = transform
    )

    train_data = torch.utils.data.DataLoader(train_datasets, batch_size = batch_size, shuffle = True)
    test_data = torch.utils.data.DataLoader(test_datasets, batch_size = batch_size, shuffle = False)

    model = get_model(name).to(device) # in PyTorch, model and data must be on the same device

    loss_function = nn.CrossEntropyLoss()                       # internally applies log-softmax
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)# Adam: adaptive learning rate, fast convergence, easy tuning

    train(model, epochs, train_data, device, loss_function, optimizer)
    test(model, test_data, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model", type=str, default="baseline")
    args = parser.parse_args()

    run(args.epochs, args.batch_size, args.model)