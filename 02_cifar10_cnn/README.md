# 02_cifar10_cnn

## Core Libraries

### torch

`torch.optim`: Module for updating model parameters (optimizers)

`torch.optim.lr_scheduler`: Module for updating the learning rate during training

### sklearn

`sklearn`: Tools for traditional machine learning and data processing

`sklearn.model_selection`: Core module for splitting datasets

`sklearn.model_selection.train_test_split()`: Function for splitting datasets

`sklearn.model_selection.StratifiedKFold (skf)`: Class for performing K-fold cross-validation while preserving class distribution

`skf.split()`: Function for returning indices corresponding to the training and validation data

### numpy

`numpy (np)`: Core library for numerical computations

`np.unique()`: Function that removes duplicates and returns unique values

`np.where()`: Function that returns indices satisfying a condition or selects values based on a condition

`np.arange()`: Function that generates numbers at regular intervals

---

## Core Concepts

**Conv / Pooling output size Formula**

$O = \lfloor \frac{I + 2P - K }{S} \rfloor + 1$

- O: Output size
- I: Input size
- P: Padding
- K: Kernel size
- S: Stride

**Training Best Practices**

- When using k-fold cross-validation, a new model and optimizer are required for each fold.
- `optimizer.zero_grad()` is required before backpropagation at each batch to prevent gradient accumulation.
- Early stopping is based on validation metrics (e.g., loss or accuracy).
- Validation is typically performed after each epoch.

--- 
## Experiment

### CIFAR10 Dataset

- Image size: 32 × 32 (RGB)
- Classes: 10
- Training data: 50,000 samples
- Test data: 10,000 samples

### Model Architecture

#### Baseline Model

| Layer       | Output Shape     | Details                     |
|-------------|------------------|-----------------------------|
| Input       | (3, 32, 32)      |                             |
| Conv1       | (16, 32, 32)     | 3x3, padding=1              |
| MaxPool     | (16, 16, 16)     | 2x2                         |
| Conv2       | (32, 16, 16)     | 3x3, padding=1              |
| MaxPool     | (32, 8, 8)       | 2x2                         |
| Flatten     | (2048)           |                             |
| FC1         | (128)            | ReLU                        |
| FC2         | (10)             |                             |

### Training setup

- Epochs: 3
- Batch size: 32

### Best Model (selected by validation accuracy)

| Metric | Value |
|--------|------|
| Epoch | 3 |
| Train Loss | 0.8974 |
| Train Accuracy | 68.33% |
| Validation Loss | 0.9125 |
| Validation Accuracy | 67.28% |

---

### Mini Task: Model Performance Improvement

#### 1. Hyperparameter Tuning (Model Architecture)

Conv layers: 2 → 3, Kernel size: 3 → 5

Feature extraction was improved by increasing model depth and kernel size.

##### Task1 model

| Layer       | Output Shape     | Details                     |
|-------------|------------------|-----------------------------|
| Input       | (3, 32, 32)      |                             |
| Conv1       | (16, 32, 32)     | 5x5, padding=2              |
| MaxPool     | (16, 16, 16)     | 2x2                         |
| Conv2       | (32, 16, 16)     | 5x5, padding=2              |
| MaxPool     | (32, 8, 8)       | 2x2                         |
| Conv3       | (64, 8, 8)       | 5x5, padding=2              |
| MaxPool     | (64, 4, 4)       | 2x2                         |
| Flatten     | (1024)           |                             |
| FC1         | (128)            | ReLU                        |
| FC2         | (10)             |                             |

##### Best Model (selected by validation accuracy)

| Metric | Value |
|--------|------|
| Epoch | 3 |
| Train Loss | 0.8639 |
| Train Accuracy | 69.65% |
| Validation Loss | 0.9045 |
| Validation Accuracy | 67.90% |


#### 2. Hyperparameter Tuning (Training Setup)

Epoch: 3 → 10

##### Best Model (selected by validation accuracy)

| Metric | Value |
|--------|------|
| Epoch | 6 |
| Train Loss | 0.5484 |
| Train Accuracy | 80.72% |
| Validation Loss | 0.8244 |
| Validation Accuracy | 72.78% |

##### Analysis

The validation accuracy is lower than the training accuracy, suggesting that the model may be overfitting to the training data.


#### 3. Training Improvements

Several training strategies were applied to improve model performance:

- Early Stopping: epochs = 100, patience = 7
- Batch Size: 32 → 64
- Optimizer Comparison:
    - Adam
    - SGD
    - AdamW
    - SGD + Momentum

##### Best Model (AdamW, selected by validation accuracy)

| Metric | Value |
|--------|------|
| Epoch | 9 |
| Train Loss | 0.4213 |
| Train Accuracy | 85.02% |
| Validation Loss | 0.8645 |
| Validation Accuracy | 73.72% |


#### 4. Cross-Validation

Stratified k-fold cross-validation was applied to evaluate model performance across different data subsets.

- Optimizer: AdamW
- K-fold: 5

##### Results

| Metric | Value |
|--------|------|
| Validation Mean Loss | 1.0119|
| Validation Mean Accuracy | 68.91% |

##### Analysis

Since the average accuracy was used, the result is lower than the previous attempt (3.),which used the highest accuracy.


#### 5. Learning Rate Scheduling

Different learning rate schedulers were applied to evaluate and compare model performance.

- CosineAnnealingLR
    - optimizer: AdamW
    - T_max: epochs
    - eta_min: 1e-6

- OneCycleLR
    - optimizer: AdamW
    - max_lr: 0.01
    - steps_per_epoch: len(train_data)
    - epochs: epochs

##### CosineAnnealingLR Results

| Metric | Value |
|--------|------|
| Validation Mean Loss | 0.9951 |
| Validation Mean Accuracy | 68.70% |

##### OneCycleLR Results

| Metric | Value |
|--------|------|
| Validation Mean Loss | 1.0435 |
| Validation Mean Accuracy | 64.55% |

##### Comparison with Baseline

| Metric | Value |
|--------|------|
| Baseline Validation Mean Accuracy | 68.91% |
| CosineAnnealingLR Validation Mean Accuracy| 68.70% |
| OneCycleLR Validation Mean Accuracy| 64.55% |

##### Analysis

Both schedulers resulted in lower performance compared to the baseline.

This suggests that the current learning rate is already well-tuned.

---