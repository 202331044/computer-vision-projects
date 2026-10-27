import torch
import numpy as np
import sys
from sklearn.metrics import average_precision_score

def mAP(
    grid_size,
    predictions,
    ground_truths,
    num_classes,
    iou_threshold
):

    ap_values = []
    for class_idx in range(num_classes):
        ap_score = ap(
            grid_size,
            predictions,
            ground_truths,
            class_idx,
            iou_threshold
        )

        if not np.isnan(ap_score):
            ap_values.append(ap_score)
    
    if len(ap_values) == 0:
        return np.nan
    
    return float(np.mean(ap_values))


def ap(
    grid_size,
    predictions,
    ground_truths,
    class_idx,
    iou_threshold
):
    
    """
    Calculate Average Precision (AP) for a specific class.

    Args:
        predictions (list[dict]): NMS-filtered predicted detections.
            Each dictionary contains:
                - filename (str): Name of the image file.
                - box (list): Predicted bounding box in the format
                [row, col, x, y, w, h].
                - score (float): Detection confidence score multiplied
                by the predicted class probability.
                - cls (int): Predicted class ID.

        ground_truths (list[dict]): Ground-truth bounding boxes.
            Each dictionary contains:
                - filename (str): Name of the image file.
                - box (list): Ground-truth bounding box in the format
                [row, col, x, y, w, h].
                - cls (int): Ground-truth class ID.

    Returns:
        float: Average Precision (AP) for the specified class.
        Returns NaN if no ground-truth objects exist for the class,
        or 0.0 if ground-truth objects exist but no predictions
        are available for the class.
    """

    gt_by_filename = {}

    # Group ground-truth boxes by filename for the specified class
    for gt in ground_truths:
        if gt['cls'] != class_idx:
            continue
        
        gt_by_filename.setdefault(gt['filename'], []).append(gt['box'])
        
    
    num_gt = sum(len(boxes) for boxes in gt_by_filename.values())

    # Return NaN if there are no ground-truth boxes for this class
    if num_gt == 0:
        return np.nan
    
    # Select predictions belonging to the specified class
    # and sort them by confidence score in descending order
    preds_by_score = []

    for pred in predictions:
        if pred['cls'] == class_idx:
            preds_by_score.append(pred)
    
    preds_by_score.sort(key = lambda pred: pred['score'], reverse = True)

    if len(preds_by_score) == 0:
        return 0.0

    # Initialize true-positive (TP) and false-positive (FP) arrays
    # for predictions sorted by confidence score
    tp = torch.zeros(len(preds_by_score))
    fp = torch.zeros(len(preds_by_score))

    # Track whether each ground-truth box has already been matched
    matched = {
        filename: np.zeros(len(boxes), dtype = bool) 
         for filename, boxes in gt_by_filename.items()
    }

    for i in range(len(preds_by_score)):
        pred = preds_by_score[i] 
        filename = pred['filename']

        # If the image has no ground-truth boxes for this class,
        # classify the prediction as a false positive
        if filename not in gt_by_filename:
            fp[i] = 1
            continue

        # unmatched_indices = np.where(~matched[filename])[0]

        # if len(unmatched_indices) == 0:
        #     fp[i] = 1
        #     continue
        

        pred_box = torch.as_tensor(
            pred['box'],
            dtype = torch.float32
        )

        gt_boxes = torch.as_tensor(
             gt_by_filename[filename],
             dtype = torch.float32
        )

        # gt_boxes = torch.as_tensor(
        #     gt_by_filename[filename],
        #     dtype = torch.float32
        # )[unmatched_indices]

         # Calculate IoU between the prediction and all ground-truth boxes
        # Select the ground-truth box with the highest IoU
        iou_scores = iou(grid_size, pred_box, gt_boxes)
        idx = int(iou_scores.argmax().item())
        
        # If the highest IoU is below the threshold, classify as FP
        if iou_scores[idx].item() < iou_threshold:
            fp[i] = 1
        elif not matched[filename][idx]:
            matched[filename][idx] = True
            tp[i] = 1
        # If the best-matching ground-truth box has already been matched,
        # classify the prediction as FP (duplicate detection)
        else:
            fp[i] = 1

        # if iou_scores[idx].item() >= iou_threshold:
        #     matched[filename][unmatched_indices[idx]] = True
        #     tp[i] = 1
        # else:
        #     fp[i] = 1

    # Calculate recall and precision at each prediction
    tp_cumsum = torch.cumsum(tp, dim = 0)
    fp_cumsum = torch.cumsum(fp, dim = 0)

    recall = tp_cumsum / num_gt
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

    # Add boundary values to the recall and precision curves
    mrec = torch.cat([
        torch.tensor([0.0]), recall, torch.tensor([1.0])
    ])

    mpre = torch.cat([
        torch.tensor([0.0]), precision, torch.tensor([0.0])
    ])

    # Construct the precision envelope by propagating
    # the maximum precision value from right to left
    for i in range(len(mrec) - 2, -1, -1):
        mpre[i] = torch.maximum(mpre[i], mpre[i + 1])
    
    # Find indices where recall changes
    indices = torch.where(mrec[1:] != mrec[:-1])[0]

    ap = torch.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])

    return ap.item()

def NMS(
    grid_size, 
    prediction, 
    score_threshold, 
    iou_threshold,
    num_boxes, 
    num_classes,
):

    """
    Apply non-maximum suppression (NMS) to a single image.

    Args:
        prediction (Tensor):
            Shape: [grid_size, grid_size, num_boxes * 5 + num_classes].
        score_threshold: Minimum score required to keep a detection.
        iou_threshold: Maximum IoU allowed between retained detections.

    Returns:
        list: NMS detections in the format
            [row, col, x, y, w, h, box_idx, score, class_idx].

        score: Confidence score multiplied by class probability.
    """

    with torch.no_grad():
        # Filter detections by score for each class
        filtered_detections = [[] for _ in range(num_classes)]

        for row in range(grid_size):
            for col in range(grid_size):
                class_probs = prediction[row, col, num_boxes * 5:]
                class_idx = class_probs.argmax().item()
                class_prob = class_probs[class_idx].item()

                for box_idx in range(num_boxes):
                    start = box_idx * 5
                    x, y, w, h, confidence = prediction[
                        row, col, start: start + 5
                    ].tolist()
                    
                    score = confidence * class_prob

                    if score > score_threshold:
                        filtered_detections[class_idx].append([
                            row, 
                            col,
                            x,
                            y,
                            w,
                            h,
                            box_idx,
                            score,
                            class_idx,
                        ])

        # Apply NMS independently to each class
        keep = []

        for detections in filtered_detections:
            detections.sort(key = lambda x : x[7], reverse=True)

            while detections:
                best_detection = detections.pop(0)
                keep.append(best_detection)

                row, col, x, y, w, h = best_detection[:6]
                remaining = []

                pred_box = torch.tensor([row, col, x, y, w, h])

                for detection in detections:
                    cmp_row, cmp_col, cmp_x, cmp_y, cmp_w, cmp_h = (
                        detection[:6]
                    )

                    tgt_box = torch.tensor([
                        cmp_row, cmp_col, cmp_x, cmp_y, cmp_w, cmp_h
                    ])

                    iou_score = iou(
                        grid_size,
                        pred_box,
                        tgt_box,
                    )

                    if iou_score.item() <= iou_threshold:
                        remaining.append(detection)

                detections = remaining

        return keep

def get_predictors(targets, predictions, grid_size, num_boxes):
    
    """
    Returns the best predictor index for each grid cell in a batch.

    Args:
        targets (Tensor): Ground-truth bounding boxes.
            Shape: [batch_size, grid_size, grid_size, 6].
                Each cell contains [x, y, w, h, class_id, objectness].

        predictions (Tensor): Predicted bounding boxes.
            Shape: [batch_size, grid_size, grid_size, num_boxes * 5 + num_classes].
                Each box contains [x, y, w, h, confidence],
                followed by class scores.

    Returns:
        Tensor: Shape [batch_size, grid_size, grid_size].
            Each element contains the index of the predictor with the highest IoU
            for the corresponding occupied cell. Empty cells are assigned -1.
    """

    with torch.no_grad():
        batch_size = targets.size(0)
        batch_predictors = []

        pred_boxes = predictions[..., : num_boxes * 5].reshape(
            batch_size, 
            grid_size,
            grid_size,
            num_boxes,
            5,
        )[..., :4]

        target_boxes = targets[..., :4]

        for batch_idx in range(batch_size):
            img_predictor = torch.full(
                (grid_size, grid_size),
                -1,
                dtype=torch.long,
                device=targets.device
            )   

            for row in range(grid_size):
                for col in range(grid_size):
                    if targets[batch_idx, row, col, 5] == 0:
                        continue

                    # Add row and col to match iou()'s input format.
                    pred_xywh = pred_boxes[batch_idx, row, col]
                    target_xywh = target_boxes[batch_idx, row, col]

                    pred_box = torch.cat([
                        torch.tensor(
                            [row, col],
                            dtype = pred_xywh.dtype,
                            device = pred_xywh.device,
                        ).expand(num_boxes, 2),
                        pred_xywh
                    ], dim = -1)

                    target_box = torch.cat([
                        torch.tensor(
                            [row, col],
                            dtype = target_xywh.dtype,
                            device = target_xywh.device,
                        ),
                        target_xywh
                    ], dim = -1)

                    iou_score = iou(
                        grid_size,
                        pred_box,
                        target_box
                    )

                    img_predictor[row, col] = iou_score.argmax(dim = 0)

            batch_predictors.append(img_predictor)

        return torch.stack(batch_predictors)

def iou(grid_size, pred_box, target_box):

    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Args:
        pred_box (Tensor): [..., 6], containing
            [row, col, x, y, w, h].
            x and y are relative to the grid cell.
            w and h are normalized by the image dimensions.
        target_box (Tensor): [..., 6], containing
            [row, col, x, y, w, h].
            Uses the same coordinate format as pred_box.

    Returns:
        Tensor: IoU score between the predicted and target boxes.
    """

    with torch.no_grad():
        pred_row, pred_col = pred_box[..., 0], pred_box[..., 1]
        pred_x, pred_y = pred_box[..., 2], pred_box[..., 3]
        pred_w, pred_h = pred_box[..., 4], pred_box[..., 5]

        tgt_row, tgt_col = target_box[..., 0], target_box[..., 1]
        tgt_x, tgt_y = target_box[..., 2], target_box[..., 3]
        tgt_w, tgt_h = target_box[..., 4], target_box[..., 5]

        # Convert cell-relative centers to image-relative coordinates.
        pred_x = (pred_col + pred_x) / grid_size
        pred_y = (pred_row + pred_y) / grid_size
        tgt_x = (tgt_col + tgt_x) / grid_size
        tgt_y = (tgt_row + tgt_y) / grid_size

        # Convert center coordinates to corner coordinates.
        pred_x1 = pred_x - pred_w / 2
        pred_y1 = pred_y - pred_h / 2
        pred_x2 = pred_x + pred_w / 2
        pred_y2 = pred_y + pred_h / 2

        tgt_x1 = tgt_x - tgt_w / 2
        tgt_y1 = tgt_y - tgt_h / 2
        tgt_x2 = tgt_x + tgt_w / 2
        tgt_y2 = tgt_y + tgt_h / 2
        
        # Calculate the intersection area.
        x1 = torch.maximum(pred_x1, tgt_x1)
        x2 = torch.minimum(pred_x2, tgt_x2)
        y1 = torch.maximum(pred_y1, tgt_y1)
        y2 = torch.minimum(pred_y2, tgt_y2)

        inter_w = torch.clamp(x2 - x1, min = 0)
        inter_h = torch.clamp(y2 - y1, min = 0)
        intersection = inter_w * inter_h

        # Calculate the union area.
        pred_area = pred_w * pred_h
        tgt_area = tgt_w * tgt_h
        union = (pred_area + tgt_area - intersection).clamp(min = 1e-6)
    
        return intersection / union

def yolo_loss(
    targets, 
    predictions, 
    predictors,
    grid_size, 
    num_boxes, 
):

    """
    Computes the YOLOv1 loss for a batch of images.

    Args:
        targets (Tensor):
            Ground-truth annotations.
            Shape: [batch_size, grid_size, grid_size, 6].
            Each grid cell contains:
                [x, y, w, h, class_id, objectness]

        predictions (Tensor):
            Model predictions.
            Shape: [batch_size, grid_size, grid_size,
                    num_boxes * 5 + num_classes].
            Each bounding box predictor contains:
                [x, y, w, h, confidence]

        predictors (Tensor):
            Index of the responsible bounding box predictor
            for each grid cell.
            Shape: [batch_size, grid_size, grid_size].
            - For cells containing an object, stores the index
              of the predictor with the highest IoU.
            - For empty cells, stores -1.

    Loss Components:
        loss1:
            Squared error between predicted and target
            bounding box center coordinates (x, y).

        loss2:
            Squared error between the square roots of the
            predicted and target bounding box dimensions (w, h).

        loss3:
            Squared error between the predicted confidence
            and the IoU of the responsible predictor with
            the ground-truth bounding box.

        loss4:
            Squared confidence penalty for predictors that
            are not responsible for detecting an object.
            Includes all predictors in empty cells.

        loss5:
            Sum of squared errors between the predicted
            class scores and one-hot encoded target classes.

    Loss Weights:
        lambda_coord (float):
            Weight applied to the bounding box coordinate
            and dimension losses. Default: 5.

        lambda_noobj (float):
            Weight applied to the confidence loss of
            predictors that are not responsible for objects.
            Default: 0.5.

    Returns:
        Tensor:
            Scalar tensor containing the total loss,
            divided by the batch size.
    """

    lambda_coord = 5
    lambda_noobj = 0.5

    loss1 = 0.0
    loss2 = 0.0
    loss3 = 0.0
    loss4 = 0.0
    loss5 = 0.0

    batch_size = targets.shape[0]

    tgt_sets = []

    for batch_idx in range(batch_size):
        tgt_set = set()

        for row in range(grid_size):
            for col in range(grid_size):
                (tgt_x, tgt_y, tgt_w, tgt_h, tgt_class_id, objectness) = targets[
                    batch_idx, row, col
                ]
                
                tgt_class_id = int(tgt_class_id)
                objectness = int(objectness)

                if objectness == 0: 
                    continue

                predictor = int(predictors[batch_idx, row, col])
                pred_bbox_idx = predictor * 5

                pred_x, pred_y, pred_w, pred_h, pred_conf = predictions[
                    batch_idx, 
                    row, 
                    col, 
                    pred_bbox_idx : pred_bbox_idx + 5
                ]

                probabilities = predictions[
                    batch_idx, 
                    row, 
                    col, 
                    num_boxes * 5:
                ]
                
                pred_box = torch.tensor([
                    row, col, pred_x, pred_y, pred_w, pred_h
                ])
                tgt_box = torch.tensor([
                    row, col, tgt_x, tgt_y, tgt_w, tgt_h
                ])
                tgt_conf = iou(
                    grid_size,
                    pred_box,
                    tgt_box,
                ).detach()


                loss1 += ((tgt_x - pred_x) ** 2 + (tgt_y - pred_y) ** 2)
                loss2 += ((torch.sqrt(tgt_w) - torch.sqrt(pred_w)) ** 2 +
                          (torch.sqrt(tgt_h) - torch.sqrt(pred_h)) ** 2)
                loss3 += (tgt_conf - pred_conf) ** 2

                target_probs = torch.zeros_like(probabilities)
                target_probs[tgt_class_id] = 1
                loss5 += ((target_probs - probabilities) ** 2).sum()

                tgt_set.add((row, col, predictor))

        tgt_sets.append(tgt_set)

    for batch_idx in range(batch_size):
        for row in range(grid_size):
            for col in range(grid_size):
                for box_idx in range(num_boxes):
                    if (row, col, box_idx) not in tgt_sets[batch_idx]:
                        idx = box_idx * 5
                        conf = predictions[batch_idx, row, col, idx + 4]
                        loss4 += conf ** 2
                
    loss1 *= lambda_coord
    loss2 *= lambda_coord
    loss4 *= lambda_noobj

    total_loss = loss1 + loss2 + loss3 + loss4 + loss5
    
    return total_loss / batch_size