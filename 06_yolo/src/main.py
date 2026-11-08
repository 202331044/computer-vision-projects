import torch
import argparse
from dataset import VOCDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from train import train_model, evaluate
from model import YOLO
from loss import mAP
from plot import visualize_voc_predictions
from torch.utils.data import ConcatDataset

import random
import os
import tarfile
import urllib.request

def run(epochs, score_th, iou_th, map_iou):

    """
    Train and validate the YOLO model on the PASCAL VOC 2007 and 2012 dataset.

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
    EPOCHS = epochs
    BATCH_SIZE = 32
    NUM_BOXES = 2
    NUM_CLASSES = 20
    GRID_SIZE = 7
    TARGET_WIDTH = 448
    TARGET_HEIGHT = 448
    LEARNING_RATE = 2e-5
    SCORE_THRESHOLD = score_th
    IOU_THRESHOLD = iou_th
    MAP_IOU_THRESHOLD = map_iou

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.ToTensor()

    model = YOLO(GRID_SIZE, NUM_BOXES, NUM_CLASSES).to(device)

    # model.load_state_dict(
    #     torch.load(
    #         "best_model.pth",
    #         map_location = device,
    #         weights_only = True,
    #     )
    # )

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

    train_2007_dataset = VOCDataset(
        year = '2007',
        image_set = "train",
        target_width = TARGET_WIDTH, 
        target_height = TARGET_HEIGHT, 
        grid_size = GRID_SIZE,
        transform = transform,
    )
    
    train_2012_dataset = VOCDataset(
        year = '2012',
        image_set = "train",
        target_width = TARGET_WIDTH, 
        target_height = TARGET_HEIGHT, 
        grid_size = GRID_SIZE,
        transform = transform,
    )

    val_2007_dataset =  VOCDataset(
        year = '2007',
        image_set = "val",
        target_width = TARGET_WIDTH,
        target_height = TARGET_HEIGHT,
        grid_size = GRID_SIZE,
        transform = transform,
    )

    val_2012_dataset =  VOCDataset(
        year = '2012',
        image_set = "val",
        target_width = TARGET_WIDTH,
        target_height = TARGET_HEIGHT,
        grid_size = GRID_SIZE,
        transform = transform,
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
        batch_size = BATCH_SIZE,
        shuffle = True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size = BATCH_SIZE,
        shuffle = False,
    )

    # print("Start evaluate")  
    # val_loss, ground_truths, predictions = evaluate(
    #         model, 
    #         val_loader, 
    #         device, 
    #         GRID_SIZE, 
    #         NUM_BOXES, 
    #         NUM_CLASSES,
    #         SCORE_THRESHOLD, 
    #         IOU_THRESHOLD,
    #     )

    # print(f"IoU Threshold:{IOU_THRESHOLD}")
    # print(f"SCORE Threshold:{SCORE_THRESHOLD}")
    # print(f"Validation Loss:{val_loss:.4f}")

    # print("Size after NMS:", len(predictions))

    # map_score = mAP(
    #     GRID_SIZE,
    #     predictions,
    #     ground_truths,
    #     NUM_CLASSES,
    #     iou_threshold = map_iou
    # )

    # print(f"mAP IoU Threshold:{map_iou}")
    # print(f"mAP: {map_score:.4f}")

    
    # random.seed(42)

    # image_filenames = [
    #     os.path.basename(path)
    #     for path in val_dataset.dataset.images
    # ]


    # sample_images = random.sample(
    #     image_filenames,
    #     min(10, len(image_filenames))
    # )

    # for filename in sample_images:
    #     visualize_voc_predictions(
    #         dataset=val_dataset,
    #         image_filename=filename,
    #         ground_truths=ground_truths,
    #         predictions=predictions,
    #         grid_size=GRID_SIZE,
    #         class_names=val_dataset.VOC_CLASSES,
    #     )

    train_model(
        model, 
        epochs, 
        train_loader, 
        val_loader, 
        device, 
        optimizer,
        GRID_SIZE, 
        NUM_BOXES, 
        NUM_CLASSES,
        SCORE_THRESHOLD,
        IOU_THRESHOLD,
        MAP_IOU_THRESHOLD
    )

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default = 1)
    parser.add_argument("--score_th", type = float, default = 0.4)
    parser.add_argument("--iou_th", type = float, default = 0.5)
    parser.add_argument("--map_iou", type = float, default = 0.5)
    args = parser.parse_args()
    run(args.epochs, args.score_th, args.iou_th, args.map_iou)