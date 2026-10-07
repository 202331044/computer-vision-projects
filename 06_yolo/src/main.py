from dataset import VOCDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
import matplotlib.pyplot as plt
import torch
from loss import get_predictors
from train import train_model
from model import YOLO

if __name__ == '__main__':

    batch_size = 32
    num_boxes = 2
    num_classes = 20
    grid_size = 7

    image_set = 'train'
    target_width = 448
    target_height = 448

    transform = transforms.ToTensor()

    train_dataset = VOCDataset(image_set,
                               target_width, 
                               target_height, 
                               grid_size,
                               transform)

    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size,
                              shuffle = True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO(grid_size, num_boxes, num_classes)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    epochs = 3
    val_loader = []

    train_model(model, epochs, 
                train_loader, val_loader, 
                device, optimizer,
                grid_size, num_boxes, num_classes)
