import torch
from loss import yolo_loss, get_predictors, NMS, mAP

def train(
    model, 
    train_loader, 
    device, 
    optimizer,
    grid_size,
    num_boxes, 
    num_classes,
):

    """
    Train the YOLO model for one epoch.

    Args:
        model: YOLO model to train.
        train_loader: DataLoader for the training dataset.
        device: Device used for training (CPU or CUDA).
        optimizer: Optimizer used to update model parameters.
        grid_size: Number of grid cells along each image dimension.
        num_boxes: Number of bounding boxes predicted per grid cell.
        num_classes: Number of object classes.

    Returns:
        Average training loss weighted by batch size.
    """

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
    model, 
    val_loader, 
    device, 
    grid_size, 
    num_boxes, 
    num_classes,
    score_threshold, 
    iou_threshold,
):

    """
    Evaluate the YOLO model on the validation dataset.

    Args:
        model: YOLO model to evaluate.
        val_loader: DataLoader for the validation dataset.
        device: Device used for evaluation (CPU or CUDA).
        grid_size: Number of grid cells along each image dimension.
        num_boxes: Number of bounding boxes predicted per grid cell.
        num_classes: Number of object classes.
        score_threshold: Minimum confidence score for detections.
        iou_threshold: IoU threshold used for non-maximum suppression (NMS).

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
        - average_loss (float): Average validation loss.

        - ground_truths (list[dict]): Ground-truth bounding boxes collected
        from the target grid. Each dictionary contains:
            - filename (str): Name of the image file.
            - box (list): Ground-truth bounding box in the format
            [row, col, x, y, w, h].
            - cls (int): Ground-truth class ID.

        - results (list[dict]): NMS-filtered predicted detections.
        Each dictionary contains:
            - filename (str): Name of the image file.
            - box (list): Predicted bounding box in the format
            [row, col, x, y, w, h].
            - score (float): Detection confidence score multiplied
            by the predicted class probability.
            - cls (int): Predicted class ID.
    """

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
                            'box': [row, col, x, y, w, h],
                            'cls': class_id
                        })

                # Apply non-maximum suppression (NMS)
                nms_results = NMS(
                    grid_size,
                    prediction,
                    score_threshold,
                    iou_threshold,
                    num_boxes,
                    num_classes,
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
    model, 
    epochs, 
    train_loader,
    val_loader, 
    device, 
    optimizer,
    grid_size, 
    num_boxes,
    num_classes,
    score_threshold,
    iou_threshold,
):
    
    best_loss = float('inf')

    for epoch in range(epochs):  
        print(f"\nEpoch [{epoch + 1}/{epochs}]")

        train_loss = train(
            model, 
            train_loader, 
            device, 
            optimizer, 
            grid_size, 
            num_boxes, 
            num_classes,
        )

        val_loss, ground_truths, predictions = evaluate(
            model, 
            val_loader, 
            device, 
            grid_size, 
            num_boxes, 
            num_classes,
            score_threshold, 
            iou_threshold,
        )

        print(f"Train Loss:{train_loss:.4f}")
        print(f"Validation Loss:{val_loss:.4f}")
        
        if best_loss > val_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Best model saved (val_loss: {best_loss:.4f})")

        map_score = mAP(
            grid_size,
            predictions,
            ground_truths,
            num_classes,
            iou_threshold = 0.5
        )

        print(f"mAP: {map_score:.4f}")


        