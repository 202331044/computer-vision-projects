import torch
import numpy as np

def get_predictors(targets, predictions, grid_size, num_boxes):
    
    with torch.no_grad():
        batch_predictors = []

        for batch_idx in range(targets.shape[0]):
            img_predictor = torch.zeros(grid_size, 
                                        grid_size,
                                        dtype=torch.long,
                                        device=targets.device)

            for row in range(grid_size):
                for col in range(grid_size):
                    (tgt_x, tgt_y, 
                    tgt_w, tgt_h, 
                    _, objectness) = targets[batch_idx, row, col]

                    objectness = int(objectness)

                    if objectness == 0:
                        continue

                    best_iou = -1
                    best_predictor = 0
                    
                    for box_idx in range(num_boxes):
                        start = box_idx * 5
                        (pred_x, pred_y,
                        pred_w, pred_h) = predictions[batch_idx, row, col, start:start+4]
                        
                        iou_score = iou(row,
                                        col,
                                        grid_size,
                                        (pred_x, pred_y, pred_w, pred_h),
                                        (tgt_x, tgt_y, tgt_w, tgt_h))

                        if iou_score > best_iou:
                            best_iou = iou_score
                            best_predictor = box_idx 
                        
                    img_predictor[row, col] = best_predictor

            batch_predictors.append(img_predictor)

    return torch.stack(batch_predictors)


def iou(row, col, grid_size, pred_box, target_box):

    """
    x and y are normalized relative to the grid cell,
    while w and h are normalized relative to the entire image.
    """

    pred_x, pred_y, pred_w, pred_h = pred_box
    target_x, target_y, target_w, target_h = target_box

    pred_x = (col + pred_x) / grid_size
    pred_y = (row + pred_y) / grid_size

    target_x = (col + target_x) / grid_size
    target_y = (row + target_y) / grid_size

    pred_x1 = pred_x - pred_w/2
    pred_y1 = pred_y - pred_h/2
    pred_x2 = pred_x + pred_w/2
    pred_y2 = pred_y + pred_h/2

    target_x1 = target_x - target_w/2
    target_y1 = target_y - target_h/2
    target_x2 = target_x + target_w/2
    target_y2 = target_y + target_h/2
    
    x1 = torch.maximum(pred_x1, target_x1)
    x2 = torch.minimum(pred_x2, target_x2)
    y1 = torch.maximum(pred_y1, target_y1)
    y2 = torch.minimum(pred_y2, target_y2)

    inter_w = torch.clamp(x2 - x1, min = 0)
    inter_h = torch.clamp(y2 - y1, min = 0)
    intersection = inter_w * inter_h

    union = (pred_w * pred_h + target_w * target_h) - intersection
    union = torch.clamp(union, min = 1e-6)
    
    return intersection / union

def yolo_loss(targets, predictions, predictors, 
              grid_size, num_boxes, num_classes):

    coord = 5
    noobj = 0.5

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

                (tgt_x, tgt_y, 
                 tgt_w, tgt_h, 
                 tgt_class_id, objectness) = targets[batch_idx, row, col]
                
                tgt_class_id = int(tgt_class_id)
                objectness = int(objectness)

                if objectness == 0: 
                    continue

                predictor = int(predictors[batch_idx, row, col])
                pred_idx = predictor * 5

                (pred_x, pred_y,
                 pred_w, pred_h, 
                 pred_conf) = predictions[batch_idx, 
                                          row, 
                                          col, 
                                          pred_idx : pred_idx + 5]

                tgt_conf = iou(row, col, grid_size,
                               (pred_x, pred_y, pred_w, pred_h),
                               (tgt_x, tgt_y, tgt_w, tgt_h)).detach()
        
                loss1 += ((tgt_x - pred_x) ** 2 + (tgt_y - pred_y) ** 2)

                loss2 += ((torch.sqrt(tgt_w) - torch.sqrt(pred_w)) ** 2 +
                          (torch.sqrt(tgt_h) - torch.sqrt(pred_h)) ** 2)
                
                loss3 += (tgt_conf - pred_conf) ** 2

                tgt_set.add((row, col, predictor))

                for class_idx in range(num_classes):
                    if tgt_class_id == class_idx:
                        tgt_prob = 1
                    else:
                        tgt_prob = 0

                    pred_prob = predictions[batch_idx, 
                                            row, 
                                            col,
                                            num_boxes * 5 + class_idx]

                    loss5 += (tgt_prob - pred_prob) ** 2


        tgt_sets.append(tgt_set)

    for batch_idx in range(batch_size):
        for row in range(grid_size):
            for col in range(grid_size):
                for box_idx in range(num_boxes):
                    if (row, col, box_idx) not in tgt_sets[batch_idx]:
                        idx = box_idx * 5
                        conf = predictions[batch_idx, row, col, idx + 4]
                        loss4 += conf ** 2
                
    loss1 *= coord
    loss2 *= coord
    loss4 *= noobj

    total_loss = loss1 + loss2 + loss3 + loss4 + loss5

    return total_loss

def calc_ap(detecions, pred, targets, target_class_idx, iou_th):

    detecions.sort(key = lambda x: x[-1], reverse=True)

    isValid = [True for _ in range(len(targets))]

    tp = []
    fp = []

    total = 0

    for target in targets:
        if target[-1] == target_class_idx:
            total += 1

    for detection in detecions:
        r, c, box_idx, class_idx, score = detection
        if class_idx != target_class_idx:
            continue

        start = box_idx * 5
        x, y, w, h, conf = pred[r, c, start : start + 5]

        best_iou = -1
        best_idx = -1

        for id in range(len(targets)):
            if targets[id][-1] != target_class_idx:
                continue

            tgt_r, tgt_c, tgt_x, tgt_y, tgt_w, tgt_h, _, _ = \
            targets[id]

            iou_score = iou((c + x, r + y, w, h),
                            (tgt_c + tgt_x, tgt_r + tgt_y, tgt_w, tgt_h))
            if isValid[id] and best_iou < iou_score:
                best_iou = iou_score
                best_idx = id

        if best_idx == -1 or best_iou < iou_th:
            fp.append(1)
            tp.append(0)
            continue

        isValid[best_idx] = False
        tp.append(1)
        fp.append(0)

    tp = torch.cumsum(torch.tensor(tp), dim = 0)
    fp = torch.cumsum(torch.tensor(fp), dim = 0)

    precision = tp / (tp + fp + 1e-6)

    if total == 0:
        return 0
    
    recall = tp / total

    precision = precision.detach().cpu().numpy()
    recall = recall.detach().cpu().numpy()

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    idx = np.where(recall[1:] != recall[:-1])[0]

    ap = np.sum((recall[idx + 1] - recall[idx]) * precision[idx + 1])
    
    return ap

def calc_map(num_classes, detecions, pred, targets, iou_th):
    aps = []
    for class_idx in range(num_classes):
        ap = calc_ap(detecions, pred, targets, class_idx, iou_th)
        aps.append(ap)

    return np.sum(aps) / len(aps)

def NMS(batch_size, grid_size, pred, 
        score_th, iou_th,
        num_boxes, num_classes):

    result = []

    for batch in range(batch_size):
        class_cofidence = [[] for _ in range(num_classes)]

        for r in range(grid_size):
            for c in range(grid_size):
                for class_idx in range(num_classes):
                    start = num_boxes * 5
                    prob = pred[batch, r, c, start + class_idx]

                    for box_idx in range(num_boxes):
                        conf = pred[batch, r, c, box_idx * 5 + 4]
                        score = conf * prob
                        if score > score_th:
                            class_cofidence[class_idx].append([r, c, box_idx, class_idx, score])

        keep = []

        for idx in range(num_classes):
            detections = class_cofidence[idx]
            detections.sort(key = lambda x:x[4], reverse=True)

            while len(detections) > 0:
                detection = detections.pop(0)
                r, c, box_idx, _, score = detection
                keep.append(detection)

                start = box_idx * 5

                x, y, w, h = pred[batch, r, c, start : start + 4]
                x += c
                y += r

                remaining = []

                for detection in detections:
                    cmp_r, cmp_c, cmp_box_idx, _, _ = detection
                    cmp_start = cmp_box_idx * 5
                    cmp_x, cmp_y, cmp_w, cmp_h = \
                        pred[batch, cmp_r, cmp_c, cmp_start : cmp_start + 4]
                    
                    cmp_x += cmp_c
                    cmp_y += cmp_r

                    iou_score = iou((x, y, w, h), (cmp_x, cmp_y, cmp_w, cmp_h))
                    
                    if iou_score <= iou_th:
                        remaining.append(detection)

                detections = remaining

        result.append(keep)

    return result