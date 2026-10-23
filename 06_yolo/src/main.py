import torch
import argparse
from dataset import VOCDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from train import train_model
from model import YOLO

def run(epochs, score_th, iou_th):

    """
    Train and validate the YOLO model on the PASCAL VOC 2007 dataset.

    Model and training configuration:
        - Input image size: 448 × 448
        - Grid size: 7 × 7
        - Number of bounding boxes per grid cell: 2
        - Number of classes: 20
        - Batch size: 32
        - Optimizer: AdamW (learning rate = 1e-4)

    Args:
        epochs (int): Number of training epochs.
        score_th (float): Minimum detection score required to retain a
            predicted bounding box. The score is calculated as the product
            of the predicted confidence and class probability.
        iou_th (float): IoU threshold used by Non-Maximum Suppression (NMS).
            A detection is suppressed if its IoU with a higher-scoring
            detection of the same class exceeds this threshold.
    """

    batch_size = 32
    num_boxes = 2
    num_classes = 20
    grid_size = 7
    target_width = 448
    target_height = 448

    transform = transforms.ToTensor()

    train_dataset = VOCDataset( 
        image_set = "train",
        target_width = target_width, 
        target_height = target_height, 
        grid_size = grid_size,
        transform = transform,
    )

    val_dataset =  VOCDataset( 
        image_set = "val",
        target_width = target_width,
        target_height = target_height,
        grid_size = grid_size,
        transform = transform,
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size = batch_size,
        shuffle = True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size = batch_size,
        shuffle = False
    )
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO(grid_size, num_boxes, num_classes).to(device)

    model.load_state_dict(
        torch.load(
            "best_model.pth",
            map_location=device,
            weights_only=True,
        )
    )

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    train_model(
        model, 
        epochs, 
        train_loader, 
        val_loader, 
        device, 
        optimizer,
        grid_size, 
        num_boxes, 
        num_classes,
        score_th,
        iou_th
    )

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default = 1)
    parser.add_argument("--score_th", type = float, default = 0.01)
    parser.add_argument("--iou_th", type = float, default = 0.1)
    args = parser.parse_args()
    run(args.epochs, args.score_th, args.iou_th)