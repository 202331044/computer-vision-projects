import torch                                  # pytorch
import torchvision                            # image datasets + preprocessing tools
import torchvision.transforms as transforms   # preprocessing tools
import torch.nn as nn                         # layer definitions
import argparse
from utils import get_model, get_optimizer, make_train_val_data, set_seed
from train import cross_validate, run_cross_validate

def run(epochs, batch_size, model_name, opt_name, is_early_stopping, patience):
  
    set_seed(42)
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
    n_splits = 5
    
    run_cross_validate(full_train_datasets, model_name, loss_function, device, batch_size=batch_size,
                    n_splits=n_splits, epochs=epochs, patience=patience, opt_name=opt_name, 
                    is_early_stopping=is_early_stopping, load_file='splits.pkl')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model", type=str, default="baseline")
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=5)

    args = parser.parse_args()


    run(args.epochs, args.batch_size, args.model, args.optimizer, args.early_stopping, args.patience)