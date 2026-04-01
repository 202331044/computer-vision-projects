import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from model import CNN as BaselineCNN
from mini_tasks.task1 import CNN as Task1CNN
from train import train, evaluate

def get_model(model_name):
    if model_name == "baseline":
        return BaselineCNN()
    elif model_name == "task1":
        return Task1CNN()
    else:
        raise ValueError("Unkown model")

def get_optimizer(opt_name, model):
    if opt_name == "SGD":
        return optim.SGD(model.parameters(), lr=0.001)
    elif opt_name == "Adam":
        return optim.Adam(model.parameters(), lr=0.001)
    elif opt_name == "AdamW":
        return optim.AdamW(model.parameters(), lr=0.001, weight_decay = 1e-4)
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

def run(model_name, epochs, batch_size, opt_name):
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
    #train_datasets, val_datasets = split_train_val_data(full_train_datasets)
    
    train_indices, val_indices = train_test_split(
        np.arange(len(full_train_datasets)),
        test_size = 0.1,
        stratify = np.array(full_train_datasets.targets),
        random_state = 42)
    
    train_datasets = torch.utils.data.Subset(full_train_datasets, train_indices)
    val_datasets = torch.utils.data.Subset(full_train_datasets, val_indices)

    train_data = torch.utils.data.DataLoader(train_datasets, batch_size = batch_size, shuffle = True)
    val_data = torch.utils.data.DataLoader(val_datasets, batch_size = batch_size, shuffle = False)
    test_data = torch.utils.data.DataLoader(test_datasets, batch_size = batch_size, shuffle = False)

    model = get_model(model_name).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = get_optimizer(opt_name, model)

    patience = 7
    train(epochs, patience, device, train_data, val_data, model, loss_function, optimizer)
    evaluate(device, test_data, model, loss_function)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default = 32)
    parser.add_argument("--optimizer", type=str, default = "Adam")
    args = parser.parse_args()

    run(args.model, args.epochs, args.batch_size, args.optimizer)