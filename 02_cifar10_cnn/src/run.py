import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from train import train, evaluate, cross_validate, run_cross_validate
import utils as u
from pathlib import Path

def run(model_name, epochs, batch_size, opt_name, is_early_stopping, patience):

    u.set_seed(42)

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

    # #split train/val data
    # train_indices, val_indices = train_test_split(
    #     np.arange(len(full_train_datasets)),
    #     test_size = 0.1,
    #     stratify = np.array(full_train_datasets.targets),
    #     random_state = 42)
    
    # train_datasets = torch.utils.data.Subset(full_train_datasets, train_indices)
    # val_datasets = torch.utils.data.Subset(full_train_datasets, val_indices)

    # train_data = torch.utils.data.DataLoader(train_datasets, batch_size = batch_size, shuffle = True)
    # val_data = torch.utils.data.DataLoader(val_datasets, batch_size = batch_size, shuffle = False)

    # model = u.get_model(model_name).to(device)
    # optimizer = u.get_optimizer(opt_name, model)

    #test_data = torch.utils.data.DataLoader(test_datasets, batch_size = batch_size, shuffle = False)

    loss_function = nn.CrossEntropyLoss()
    n_splits = 5

    BASE_DIR = Path(__file__).resolve().parent.parent
    load_data_path = BASE_DIR / "data" / "splits.pkl"

    run_cross_validate(full_train_datasets, model_name, loss_function, device, batch_size=batch_size,
                    n_splits=n_splits, epochs=epochs, patience=patience, opt_name=opt_name,
                     is_early_stopping=is_early_stopping,  load_file=load_data_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", type=str, default="baseline")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default = 32)
    parser.add_argument("--optimizer", type=str, default = "Adam")
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=5)

    args = parser.parse_args()

    run(args.model, args.epochs, args.batch_size, args.optimizer, args.early_stopping, args.patience)