import torch                                  # pytorch
import torchvision                            # image datasets + preprocessing tools
import torchvision.transforms as transforms   # preprocessing tools
import torch.nn as nn                         # layer definitions
import argparse
from model import get_model, get_optimizer
from train import cross_validate

def run(epochs, batch_size, model_name, opt_name):
  
    # cuda = NVIDIA GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DL models accept only tensors, so convert image data (0~255 -> 0.0~1.0)
    transform = transforms.ToTensor()
    full_train_datasets = torchvision.datasets.MNIST (
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

    loss_function = nn.CrossEntropyLoss()
    patience = 5
    n_splits = 5

    cross_validate(full_train_datasets, model_name, loss_function, device, batch_size,
                n_splits, epochs, patience, opt_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model", type=str, default="baseline")
    parser.add_argument("--optimizer", type=str, default="Adam")
    
    args = parser.parse_args()

    run(args.epochs, args.batch_size, args.model, args.optimizer)