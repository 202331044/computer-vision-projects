import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import argparse
import numpy as np

from model import CNN as BaselineCNN
from mini_tasks.task1 import CNN as Task1CNN
from train import train, evaluate

def get_model(name):
    if name == "baseline":
        return BaselineCNN()
    elif name == "task1":
        return Task1CNN()

def run(name, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

    full_train_datasets = torchvision.datasets.CIFAR10(
        root = "./data",
        train = True,
        transform = transform,
        download = True
    )

    test_datasets = torchvision.datasets.CIFAR10(
        root = "./data",
        train = False,
        transform = transform
    )

    #split train/val data
    indices = np.arange(len(full_train_datasets))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices = indices[:45000]
    val_indices = indices[45000:]
    train_datasets = torch.utils.data.Subset(full_train_datasets, train_indices)
    val_datasets= torch.utils.data.Subset(full_train_datasets, val_indices)

    train_data = torch.utils.data.DataLoader(train_datasets, batch_size = 32, shuffle = True)
    val_data = torch.utils.data.DataLoader(val_datasets, batch_size = 32, shuffle = False)
    test_data = torch.utils.data.DataLoader(test_datasets, batch_size = 32, shuffle = False)

    model = get_model(name).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    train(epochs, device, train_data, val_data, model, loss_function, optimizer)
    evaluate(device, test_data, model, loss_function)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    run(args.model, args.epochs)