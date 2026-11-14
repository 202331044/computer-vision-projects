import torch
from dataset import VOCDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from train import train_model
from model import YOLO
from torch.utils.data import ConcatDataset
from utils import load_darknet_weights, build_optimizer
from config import Config


def run():
    """
    Train and validate the YOLO model using the
    PASCAL VOC 2007 and 2012 datasets.

    The training pipeline consists of the following steps:
        1. Initialize the YOLO model.
        2. Load pretrained Darknet weights.
        3. Build the optimizer.
        4. Load and combine VOC 2007 and 2012 datasets.
        5. Create training and validation DataLoaders.
        6. Train and validate the model.

    Training configurations and hyperparameters are
    managed separately in config.py through the Config class.

    Checkpoint management, including saving and resuming
    training, is handled by train_model().

    Returns:
        None
    """
    
    transform = transforms.ToTensor()

    model = YOLO(
        Config.GRID_SIZE, 
        Config.NUM_BOXES, 
        Config.NUM_CLASSES,
        Config.DROP_OUT
    )

    model = load_darknet_weights(
        model=model,
        weight_path=Config.WEIGHT_PATH,
        num_conv_layers=Config.NUM_PRETRAINED_CONV_LAYERS
    )

    model.to(Config.DEVICE)
    optimizer = build_optimizer(Config, model)

    train_2007_dataset = VOCDataset(
        year='2007',
        image_set="train",
        target_width=Config.TARGET_WIDTH, 
        target_height=Config.TARGET_HEIGHT, 
        grid_size=Config.GRID_SIZE,
        transform=transform,
    )
    
    train_2012_dataset = VOCDataset(
        year='2012',
        image_set="train",
        target_width=Config.TARGET_WIDTH, 
        target_height=Config.TARGET_HEIGHT, 
        grid_size=Config.GRID_SIZE,
        transform=transform,
    )

    val_2007_dataset =  VOCDataset(
        year='2007',
        image_set="val",
        target_width=Config.TARGET_WIDTH,
        target_height=Config.TARGET_HEIGHT,
        grid_size=Config.GRID_SIZE,
        transform=transform,
    )

    val_2012_dataset =  VOCDataset(
        year='2012',
        image_set="val",
        target_width=Config.TARGET_WIDTH,
        target_height=Config.TARGET_HEIGHT,
        grid_size=Config.GRID_SIZE,
        transform=transform,
    )

    train_dataset = ConcatDataset([
        train_2007_dataset,
        train_2012_dataset
    ])

    val_dataset = ConcatDataset([
        val_2007_dataset,
        val_2012_dataset
    ])

    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
    )

    train_model(
        Config,
        model, 
        train_loader, 
        val_loader, 
        optimizer,
    )

if __name__ == '__main__':
    run()