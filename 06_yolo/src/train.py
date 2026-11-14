import torch
from loss import yolo_loss, get_predictors, NMS, mAP
from utils import adjust_learning_rate
from config import Config
import os

def train(
    config,
    model, 
    train_loader,
    optimizer
):
    """
    Train the YOLO model for one epoch.

    Args:
        config (Config): Configuration class containing model settings,
            training hyperparameters, and device information.
        model (YOLO): YOLO model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        optimizer (Optimizer): Optimizer used to update model parameters.

    Returns:
        float: Average training loss weighted by batch size.
    """

    grid_size = config.GRID_SIZE
    num_boxes = config.NUM_BOXES
    device = config.DEVICE

    model.train()

    total_loss = 0.0
    total_size = 0

    for imgs, targets, filenames in train_loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        batch_size = imgs.size(0)

        optimizer.zero_grad()

        predictions = model(imgs)

        predictors = get_predictors(targets, predictions, grid_size, num_boxes)

        loss = yolo_loss(
            targets, 
            predictions, 
            predictors, 
            grid_size, 
            num_boxes, 
        )
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_size += batch_size

    return total_loss / total_size

def evaluate(
    config,
    model, 
    val_loader, 
):

    """
    Evaluate the YOLO model on the validation dataset.

    Args:
        config (Config): Configuration class containing model settings,
            evaluation thresholds, and device information.
        model (YOLO): YOLO model to evaluate.
        val_loader (DataLoader): DataLoader providing validation images
            and their corresponding targets.

    Tensor Shapes:
        predictions (Tensor):
        Shape: [batch_size, grid_size, grid_size, num_boxes * 5 + num_classes]
            Each bounding box prediction contains:
                - x, y: Predicted bounding box center coordinates.
                - w, h: Predicted bounding box width and height.
                - confidence: Predicted confidence score.
            Class scores are normalized with softmax and sum to 1
            across classes for each grid cell.

        targets (Tensor):
            Shape: [batch_size, grid_size, grid_size, 6]
            Each grid cell contains:
                - x_cell_offset, y_cell_offset: Bounding box center coordinates
                relative to the assigned grid cell, normalized to [0, 1).
                - bbox_w_ratio, bbox_h_ratio: Bounding box width and height
                normalized by the target image dimensions.
                - class_id: Integer index of the object class.
                - objectness: 1 if an object is assigned to the cell,
                otherwise 0.

    Output Details:
        detection (list):
            NMS results for a single image. Each detection contains:
                [row, col, x, y, w, h, box_idx, score, class_idx]
            score: Confidence score multiplied by class probability.

    Returns:
        tuple:
            - average_loss (float): Average validation loss weighted
                by batch size.
            - ground_truths (list[dict]): Ground-truth bounding boxes.
                Each dictionary contains:
                    - filename (str): Image filename.
                    - box (list): Bounding box in the format
                        [row, col, x, y, w, h].
                    - cls (int): Ground-truth class ID.
            - results (list[dict]): NMS-filtered predicted detections.
                Each dictionary contains:
                    - filename (str): Image filename.
                    - box (list): Predicted bounding box in the format
                        [row, col, x, y, w, h].
                    - score (float): Detection score.
                    - cls (int): Predicted class ID.
    """

    device = config.DEVICE
    grid_size = config.GRID_SIZE
    num_boxes = config.NUM_BOXES

    model.eval()

    total_loss = 0.0
    total_size = 0

    results = []
    ground_truths = []

    with torch.no_grad():

        for imgs, targets, filenames in val_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            batch_size = imgs.size(0)

            predictions = model(imgs)

            predictors = get_predictors(
                targets, 
                predictions, 
                grid_size, 
                num_boxes,
            )

            loss = yolo_loss(
                targets, 
                predictions, 
                predictors,
                grid_size, 
                num_boxes, 
            )
            
            total_loss += loss.item() * batch_size
            total_size += batch_size

            # Collect ground-truth boxes and apply NMS
            for filename, prediction, target in zip(
                filenames, 
                predictions, 
                targets
            ):

                # Extract ground-truth boxes from the target grid
                for row in range(grid_size):
                    for col in range(grid_size):
                        x, y, w, h, class_id, objectness = target[row, col]

                        if objectness.item() < 1:
                            continue

                        class_id = int(class_id.item())

                        ground_truths.append({
                            'filename': filename,
                            'box': [
                                row, 
                                col, 
                                x.item(), 
                                y.item(),
                                w.item(), 
                                h.item()
                            ],
                            'cls': class_id
                        })

                # Apply non-maximum suppression (NMS)
                nms_results = NMS(
                    config,
                    prediction
                )

                for img_result in nms_results:
                    results.append({
                        'filename' : filename, 
                        'box': img_result[:6],
                        'score': float(img_result[7]),
                        'cls': int(img_result[8])
                    })


    return total_loss / total_size, ground_truths, results

def train_model(
    config,
    model, 
    train_loader,
    val_loader,
    optimizer,
    start_epoch=0
):

    epochs = config.EPOCHS
    checkpoint_path = config.CHECKPOINT_PATH
    best_model_path = config.BEST_MODEL_PATH

    best_loss = float('inf')

    # Resume training
    # if os.path.exists(checkpoint_path):
    #     checkpoint = torch.load(
    #         checkpoint_path,
    #         map_location=config.DEVICE,
    #         weights_only=False
    #     )

    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    #     start_epoch = checkpoint['epoch'] + 1

    #     best_loss = checkpoint.get(
    #         'best_loss',
    #         float('inf')
    #     )

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch [{epoch + 1}/{epochs}]")

        adjust_learning_rate(
            config, 
            optimizer,
            epoch,
        )

        train_loss = train(
            config,
            model, 
            train_loader, 
            optimizer, 
        )

        val_loss, ground_truths, predictions = evaluate(
            config,
            model, 
            val_loader
        )

        print(f"Train Loss:{train_loss:.4f}")
        print(f"Validation Loss:{val_loss:.4f}")
        
        if best_loss > val_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved (val_loss: {best_loss:.4f})")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_loss': best_loss,
        }

        temp_path = checkpoint_path + '.tmp'

        torch.save(checkpoint, temp_path)
        os.replace(temp_path, checkpoint_path)

        map_score = mAP(
            predictions,
            ground_truths,
            config.GRID_SIZE,
            config.NUM_CLASSES,
            config.MAP_IOU_THRESHOLD
        )

        print(f"mAP: {map_score:.4f}")