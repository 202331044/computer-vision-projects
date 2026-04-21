# 02 CIFAR-10 CNN: Image Classification

## 🧰 Core Libraries

### torch

`torch.optim`: Module that optimizes model parameters

`torch.optim.lr_scheduler`: Module that adjusts the learning rate during training

`torch.nn.AdaptiveMaxPool2d`: Layer that outputs a fixed spatial size regardless of input size

`torch.nn.BatchNorm2d`: Layer that normalizes feature distributions to stabilize training and improve generalization

### sklearn

`sklearn`: Library that provides tools for traditional machine learning and data processing

`sklearn.model_selection`: Module that handles dataset splitting and evaluation

`sklearn.model_selection.train_test_split()`: Function that splits datasets into training and validation sets

`sklearn.model_selection.StratifiedKFold (skf)`: Class that performs K-fold cross-validation while preserving class distribution

`skf.split()`: Function that returns indices for training and validation data


### numpy

`numpy (np)`: Library that provides core functionality for numerical computations

`np.unique()`: Function that returns unique values by removing duplicates

`np.where()`: Function that returns indices that satisfy a condition or selects values based on a condition

`np.arange()`: Function that generates values at regular intervals

---

## 📚 Core Concepts

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

- Pooling Reordering: Applying pooling after consecutive convolution layers (Conv → Conv → Pool) to enable richer feature extraction.

**Batch Nomalization**

- Typically applied after convolution layers (Conv → BatchNorm → ReLU).

- Stabilizes output distributions and prevents gradient explosion/vanishing.

- Enables faster and more stable convergence by reducing optimization variance.

- Provides weak regularization through noise from mini-batch statistics.

--- 
## 🚀 Experiment

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

- Cross-validation: Stratified K-Fold (K=5)
- Epochs: 10
- Batch size: 32
- Optimizer: Adam

### Results

| Metric | Value |
|--------|-------|
| Validation Loss | 0.9138 ± 0.0179 |
| Validation Accuracy | 68.59% ± 0.63% |

> Results are averaged over 5 folds.

---

### Model Performance Improvement

To improve CNN performance on CIFAR-10 classification, the following experiments were conducted.

### 1. Training Stability
- Early Stopping
- Patience tuning

### 2. Regularization
- Dropout
- Batch Normalization

### 3. Architecture Changes
- Additional hidden layers
- Conv → Conv → Pool restructuring

### 4. Pooling Strategies
- Adaptive Max Pooling → Adaptive Avg Pooling
- Global Average Pooling

---