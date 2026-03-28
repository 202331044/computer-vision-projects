import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import argparse

from model import CNN as BaselineCNN
from mini_tasks.task1 import CNN as Task1CNN
from mini_tasks.task2 import CNN as Task2CNN
from train import train, test

def get_model(name):
    if name == "baseline":
        return BaselineCNN()
    elif name == "task1":
        return Task1CNN()
    elif name == "task2":
        return Task2CNN()

def run(name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

    train_datasets = torchvision.datasets.CIFAR10(
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

    train_data = torch.utils.data.DataLoader(train_datasets, batch_size = 32, shuffle = True)
    test_data = torch.utils.data.DataLoader(test_datasets, batch_size = 32, shuffle = False)

    model = get_model(name).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    train(3, device, train_data, model, loss_function, optimizer)
    test(device, test_data, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline")

    args = parser.parse_args()

    run(args.model)