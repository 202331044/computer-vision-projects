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

YOLO is a single-stage object detection model that predicts bounding boxes and class probabilities directly from an input image in a single forward pass.

The input image is divided into a grid, where each cell is responsible for detecting objects whose centers fall within it.

**Key Characteristics**
- Grid-based object detection.
- Predicts multiple bounding boxes and confidence scores per grid cell.
- Predicts class probabilities for each grid cell.

**Advantages**
- Fast inference without region proposal generation.
- Uses global image context.
- Jointly learns localization and classification.

**Disadvantages**
- Relatively high localization errors due to limited bounding box localization precision.
- Limited ability to detect multiple objects within the same grid cell.
- Struggles with small objects and groups of nearby objects.

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

### Loss Function

YOLO v1 uses a multi-part Sum of Squared Errors (SSE) loss function to jointly optimize bounding box localization, confidence prediction, and classification.

The overall loss consists of five components:
- Bounding box center coordinate loss.
- Bounding box dimension loss.
- Responsible predictor confidence loss.
- No-object confidence loss.
- Classification loss.

#### Overall Loss

The original YOLO v1 loss function is defined as:
$$
\begin{aligned}
\mathcal{L} ={}&
\lambda_{coord}
\sum_{i=0}^{S^2-1}
\sum_{j=0}^{B-1}
\mathbb{1}_{ij}^{obj}
\left[
(x_{ij}-\hat{x}_{ij})^2
+
(y_{ij}-\hat{y}_{ij})^2
\right]
\\[6pt]
&+
\lambda_{coord}
\sum_{i=0}^{S^2-1}
\sum_{j=0}^{B-1}
\mathbb{1}_{ij}^{obj}
\left[
(\sqrt{w_{ij}}-\sqrt{\hat{w}_{ij}})^2
+
(\sqrt{h}_{ij}-\sqrt{\hat{h}_{ij}})^2
\right]
\\[6pt]
&+
\sum_{i=0}^{S^2-1}
\sum_{j=0}^{B-1}
\mathbb{1}_{ij}^{obj}
(C_{ij}-\hat{C}_{ij})^2
\\[6pt]
&+
\lambda_{noobj}
\sum_{i=0}^{S^2-1}
\sum_{j=0}^{B-1}
\mathbb{1}_{ij}^{noobj}
(C_{ij}-\hat{C}_{ij})^2
\\[6pt]
&+
\sum_{i=0}^{S^2-1}
\mathbb{1}_{i}^{obj}
\sum_{c=1}^{C}
(p_i(c)-\hat{p}_i(c))^2
\end{aligned}
$$

**Indicator Functions**

Indicator functions determine which predictions contribute to each loss component.

- $\mathbb{1}_{ij}^{obj}$: Equals 1 if the (j)-th bounding box predictor in grid cell (i) is responsible for detecting an object; otherwise, 0.
- $\mathbb{1}_{ij}^{noobj}$: Equals 1 if the (j)-th bounding box predictor in grid cell (i) is not responsible for detecting an object; otherwise, 0.
- $\mathbb{1}_{i}^{obj}$: Equals 1 if grid cell (i) contains an object; otherwise, 0.

**Loss Hyperparameters**

YOLO v1 uses two weighting coefficients to balance localization and confidence losses.

- **$\lambda_{coord}=5$**: Increases the weight of bounding box localization loss, giving more importance to accurate object localization.

- **$\lambda_{noobj}=0.5$**: Reduces the weight of confidence loss for bounding box predictors that are not responsible for detecting objects, preventing background predictions from dominating the overall loss.

### Loss Components

YOLO v1 uses a multi-part loss function consisting of five components.

- **Bounding Box Center Coordinate Loss**

   Penalizes the squared error between predicted and ground-truth bounding box center coordinates.

   $$
   \mathcal{L}_{xy} =
   \lambda_{coord}
   \sum_i \sum_j
   \mathbb{1}_{ij}^{obj}
   \left[
   (x_{ij}-\hat{x}_{ij})^2
   +
   (y_{ij}-\hat{y}_{ij})^2
   \right]
   $$

- **Bounding Box Dimension Loss**

   Penalizes the squared error between the square roots of predicted and ground-truth bounding box dimensions.

   $$
   \mathcal{L}_{wh} =
   \lambda_{coord}
   \sum_i \sum_j
   \mathbb{1}_{ij}^{obj}
   \left[
   (\sqrt{w_{ij}}-\sqrt{\hat{w}_{ij}})^2
   +
   (\sqrt{h}_{ij}-\sqrt{\hat{h}_{ij}})^2
   \right]
   $$

   Using square roots reduces the relative impact of errors in large bounding boxes and gives more importance to localization accuracy for smaller boxes.

- **Responsible Predictor Confidence Loss**

   Penalizes the squared error between the predicted confidence and its target for the responsible bounding box predictor.

   $$
   \mathcal{L}_{obj} =
   \sum_i \sum_j
   \mathbb{1}_{ij}^{obj}
   (C_{ij}-\hat{C}_{ij})^2
   $$

   The target confidence is the IoU between the predicted bounding box and the ground-truth bounding box:

   $$
   {C}_{ij} =
   IoU(B_{ij}^{pred}, B_i^{GT})
   $$

- **No-Object Confidence Loss**

   Penalizes confidence predictions from bounding box predictors that are not responsible for detecting an object.

   $$
   \mathcal{L}_{noobj} =
   \lambda_{noobj}
   \sum_i \sum_j
   \mathbb{1}_{ij}^{noobj}
   (C_{ij}-\hat{C}_{ij})^2
   $$

   The target confidence for these predictors is zero.

- **Classification Loss**

  Penalizes the squared error between predicted and ground-truth class probabilities for grid cells containing objects.

  $$
  \mathcal{L}_{class} =
  \sum_i \mathbb{1}_i^{obj}
  \sum_{c \in classes}
  (p_i(c)-\hat{p}_i(c))^2
  $$

--- 

## 🏗️ Model Implementation

The YOLO model is implemented using PyTorch and follows the CNN-based architecture of YOLO v1.

### Overall Architecture

The YOLO v1 model consists of six convolutional stages followed by two fully connected layers. The convolutional layers extract visual features, while the fully connected layers generate the final object detection predictions.

| Layer | Configuration | Output |
|---|---|---|
| Conv1 | 7×7, 64 filters, stride 2, padding 3 | 64 channels |
| MaxPool1 | 2×2, stride 2 | Downsampled feature maps |
| Conv2 | 3×3, 192 filters, padding 1 | 192 channels |
| MaxPool2 | 2×2, stride 2 | Downsampled feature maps |
| Conv3 | 1×1, 128 → 3×3, 256 → 1×1, 256 → 3×3, 512 (3×3 convolutions use padding 1) | 512 channels |
| MaxPool3 | 2×2, stride 2 | Downsampled feature maps |
| Conv4 | (1×1, 256 → 3×3, 512) × 4 → 1×1, 512 → 3×3, 1024 (3×3 convolutions use padding 1) | 1024 channels |
| MaxPool4 | 2×2, stride 2 | Downsampled feature maps |
| Conv5 | (1×1, 512 → 3×3, 1024) × 2 → 3×3, 1024 → 3×3, 1024 (all 3×3 convolutions use padding 1; final layer uses stride 2) | 1024 channels |
| Conv6 | Two 3×3 convolutional layers, 1024 filters each, padding 1 | 1024 channels |
| Flatten | Flatten feature maps | 50,176 |
| FC1 | Fully connected, 4096 units + Leaky ReLU | 4096 |
| Dropout | Dropout rate: 0.5 | 4096 |
| FC2 | Fully connected | N × [S × S × (B × 5 + C)] |
| Reshape | Reshape predictions into a 4D tensor | N × S × S × (B × 5 + C) |

N denotes the batch size.

All convolutional layers use Batch Normalization and Leaky ReLU activation (slope = 0.1).

---