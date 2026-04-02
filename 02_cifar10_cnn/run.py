import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

from train_utils import train, evaluate, cross_validate
from utils import get_model, get_optimizer, split_train_val_data

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
    ##train_datasets, val_datasets = split_train_val_data(full_train_datasets)
    
    # train_indices, val_indices = train_test_split(
    #     np.arange(len(full_train_datasets)),
    #     test_size = 0.1,
    #     stratify = np.array(full_train_datasets.targets),
    #     random_state = 42)
    
    # train_datasets = torch.utils.data.Subset(full_train_datasets, train_indices)
    # val_datasets = torch.utils.data.Subset(full_train_datasets, val_indices)

    # train_data = torch.utils.data.DataLoader(train_datasets, batch_size = batch_size, shuffle = True)
    # val_data = torch.utils.data.DataLoader(val_datasets, batch_size = batch_size, shuffle = False)

    test_data = torch.utils.data.DataLoader(test_datasets, batch_size = batch_size, shuffle = False)

    # model = get_model(model_name).to(device)

    loss_function = nn.CrossEntropyLoss()
    # optimizer = get_optimizer(opt_name, model)

    patience = 7
    
    # train(epochs, patience, device, train_data, val_data, model, loss_function, optimizer)
    # evaluate(device, test_data, model, loss_function)
    
    cross_validate(full_train_datasets, epochs, patience, device, model_name, loss_function, opt_name,
        n_splits=5, batch_size=64, random_state=42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default = 32)
    parser.add_argument("--optimizer", type=str, default = "Adam")
    args = parser.parse_args()

    run(args.model, args.epochs, args.batch_size, args.optimizer)