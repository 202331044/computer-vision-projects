# 06. YOLO

Implement YOLO v1 from scratch using PyTorch and experiment with object detection on the PASCAL VOC dataset.

## 📚 Core Concepts

### Object Detection

Object Detection is a computer vision task that identifies objects in an image and predicts their locations and classes.

Unlike image classification, which assigns a single label to an entire image, object detection detects multiple objects and predicts their corresponding bounding boxes and class labels.

Each detection prediction consists of:

- **x, y:** Center coordinates of the bounding box.
- **w, h:** Width and height of the bounding box.
- **Class:** Predicted object category.
- **Confidence:** A score representing the confidence of the prediction.

Object detection methods can be broadly categorized into two types:

- **Two-stage Detectors**: First generate region proposals that are likely to contain objects, then classify the proposed regions and refine their bounding boxes.
   - R-CNN, Fast R-CNN, Faster R-CNN
- **One-stage Detectors**: Directly predict object locations and classes in a single forward pass, without a separate region proposal stage.
   - YOLO, SSD

### IoU (Intersection over Union)

IoU measures the overlap between a predicted bounding box and a ground-truth bounding box.

$$
IoU = \frac{Area\ of\ Intersection}{Area\ of\ Union}
$$

**Characteristics**

- A value of 1 indicates perfect overlap.
- A value of 0 indicates no overlap.
- Commonly used to evaluate localization accuracy.
- Used in Non-Maximum Suppression (NMS) to identify overlapping bounding boxes.

### AP (Average Precision)

AP measures object detection performance for a specific class.

It is calculated as the area under the Precision-Recall curve.

$$
AP = \int_0^1 Precision(Recall)\,dRecall
$$

- **Precision**: The ratio of correctly detected objects among all predicted objects.
- **Recall**: The ratio of correctly detected objects among all ground-truth objects.

### mAP (Mean Average Precision)

mAP is the mean of Average Precision values across all object classes.

It provides an overall measure of object detection performance across multiple classes.

$$
mAP = \frac{1}{C}\sum_{c=1}^{C} AP_c
$$

- **C**: Number of object classes.
- **$AP_{c}$**: Average Precision of class $c$.

### NMS (Non-Maximum Suppression)

NMS is a post-processing algorithm that removes redundant bounding boxes corresponding to the same object.

It is performed independently for each class using these class-specific scores.

**Process**

1. Select the bounding box with the highest class-specific score.
2. Add the selected box to the final detection results.
3. Calculate IoU between the selected box and the remaining boxes of the same class.
4. Remove boxes whose IoU exceeds the predefined threshold.
5. Repeat the process until no candidate boxes remain.

**Purpose**

- Removes duplicate detections.
- Keeps the most confident bounding box among highly overlapping predictions.
- Improves the quality of final detection results.

---

## 🤖 YOLO v1 (You Only Look Once)

### Overview

YOLO is a single-stage object detection model that predicts bounding boxes and class probabilities directly from an input image.

Unlike two-stage detectors, which first generate region proposals and then classify them, YOLO performs object localization and classification in a single forward pass.

The input image is divided into a grid, and each grid cell is responsible for detecting objects whose centers fall within that cell.

**Key Characteristics**
- Performs object detection in a single forward pass.
- Uses global image context for object localization and classification.
- Combines object localization and classification into a single end-to-end architecture.

### Architecture

YOLO v1 divides the input image into an `S` X `S` grid.

Each grid cell predicts:

- **Bounding Boxes:** `B` bounding boxes, each represented by four coordinates: `x`, `y`, `w`, and `h`.
- **Confidence Score:** The probability of an object being present multiplied by the IoU between the predicted and ground truth bounding boxes.

   $$
   Pr(Object) \times IoU_{\text{pred}}^{\text{truth}}
   $$

   Where:
   - **Pr(Object):** Probability that the bounding box contains an object.
   - **$IoU_{\text{pred}}^{\text{truth}}$:** Intersection over Union between the predicted bounding box and the ground-truth bounding box.

- **Class Probabilities:** A probability distribution over the object classes for each cell.

   $$
   Pr(Class_i \mid Object)
   $$

### Output Tensor

The final output tensor has the following shape:

$$
S \times S \times (B \times 5 + C)
$$

Where:

- `S`: Grid size.
- `B`: Number of bounding boxes per grid cell.
- `C`: Number of object classes.
- `5`: Number of bounding box parameters (`x`, `y`, `w`, `h`, confidence).

### Detection Process

1. Divide the input image into an `S` × `S` grid.
2. Each grid cell predicts `B` bounding boxes, their confidence scores, and conditional class probabilities for all object classes.
3. Calculate class-specific confidence scores for each bounding box.
4. Apply Non-Maximum Suppression (NMS) to remove redundant detections.
5. Generate the final detection results.

### Responsible Predictor

When multiple bounding box predictors are assigned to the same grid cell, only one predictor is responsible for detecting a particular object.

The responsible predictor is selected based on the highest IoU with the ground-truth bounding box.

This allows the model to learn which predictor is better suited for detecting the object.

Predictors that are not responsible for an object are assigned a confidence target of zero.

### Class-specific Confidence Score

YOLO v1 combines the conditional class probabilities with the bounding box confidence scores to calculate class-specific confidence scores.

These scores are used to rank bounding boxes and determine which detections should be retained during Non-Maximum Suppression (NMS).

$$
Score_{ij} =
Pr(Class_i \mid Object) \times Confidence_j
$$

Substituting the confidence definition:

$$
Score_{ij} =
Pr(Class_i \mid Object)
\times Pr(Object_j)
\times IoU_j
$$

Where:
- `i`: Object class index.
- `j`: Bounding box predictor index.

These scores represent the probability that a bounding box contains a specific object class, weighted by its localization accuracy.

---