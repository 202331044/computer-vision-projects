from loss import yolo_loss, iou, get_predictors
import torch
from loss import NMS

def train(model, train_loader, device, optimizer,
          grid_size, num_boxes, num_classes):
    """
    `predictions` has the shape
    [grid_size, grid_size, num_boxes * 5 + num_classes].

    The last dimension consists of
    num_boxes * [x, y, w, h, confidence],
    followed by the class scores.
    """

    model.train()

    total_loss = 0.0
    total_size = 0

    for imgs, targets in train_loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        batch_size = targets.shape[0]

        optimizer.zero_grad()

        predictions = model(imgs)
        predictions = predictions.view(-1, 
                                       grid_size, 
                                       grid_size, 
                                       num_boxes * 5 + num_classes)

        predictors = get_predictors(targets, predictions, grid_size, num_boxes)

        loss = yolo_loss(targets, predictions, predictors, 
                         grid_size, num_boxes, num_classes)
        
        total_loss += loss.item()
        total_size += batch_size

        loss.backward()
        optimizer.step()

    return total_loss / total_size

def evaluate(model, val_loader, device, 
             grid_size, num_boxes, num_classes,
             score_threshold=0.2, iou_threshold=0.5):
    model.eval()

    total_loss = 0.0
    total_size = 0
    all_detections = []

    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            batch_size = targets.shape[0]
            predictions = model(imgs)
            predictions = predictions.view(-1, 
                                           grid_size, 
                                           grid_size, 
                                           num_boxes * 5 + num_classes)

            predictors = get_predictors(targets, 
                                        predictions, 
                                        grid_size, 
                                        num_boxes)

            total_size += batch_size

            loss = yolo_loss(targets, predictions, predictors, 
                             grid_size, num_boxes, num_classes)
            total_loss += loss.item()

            detections = NMS(batch_size, grid_size, predictions, 
                             score_threshold, iou_threshold,
                             num_boxes, num_classes)
            all_detections.extend(detections)

    return total_loss / total_size, all_detections

def train_model(model, epochs, train_loader, val_loader, 
                device, optimizer,
                grid_size, num_boxes, num_classes):
    
    for epoch in range(epochs):
        train_loss = train(model, train_loader, device, optimizer, 
                           grid_size, num_boxes, num_classes)
        
        # val_loss, all_detections  = evaluate(model, val_loader, device, 
        #                     grid_size, num_boxes, num_classes)

        
        print(f"Epoch:{epoch + 1}")
        print(f"Train Loss:{train_loss:.4f}")
        #print(f"Validation Loss:{val_loss:.4f}")