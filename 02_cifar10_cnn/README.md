# 02_cifar10_cnn

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

### Mini Task: Model Performance Improvement

#### step 1. Early Stopping

- Add early stopping (Epochs: 10 → 100, patience=5)

##### Results

| Metric | Baseline | Step 1 |
|--------|----------|--------|
| Validation Loss | 0.9138 ± 0.0179 | 0.9138 ± 0.0179 |
| Validation Accuracy | 68.59% ± 0.63% | 68.59% ± 0.63% |

> Results are averaged over 5 folds.

- The results for Step 1 are identical to Baseline, suggesting that 10 epochs may already be sufficient for this setting.

#### step 2. Increase Early Stopping Patience

- Increase patience: 5 → 10

##### Results

| Metric | Baseline & Step 1 | Step 2 |
|--------|-------------------|--------|
| Validation Loss | 0.9138 ± 0.0179 | 0.9138 ± 0.0179 |
| Validation Accuracy | 68.59% ± 0.63% | 68.59% ± 0.63% |

> Results are averaged over 5 folds.

- The results for Step 2 are identical to the Baseline and Step 1, indicating no performance improvement.

- The training accuracy is significantly higher than the validation accuracy, indicating overfitting.

#### step 3. Dropout

- Add dropout (p=0.5)
- patience = 5

##### Results

| Metric | Baseline& Step 1, 2 | Step 3 |
|--------|---------------------|--------|
| Validation Loss | 0.9138 ± 0.0179 | 0.9019 ± 0.0085 |
| Validation Accuracy | 68.59% ± 0.63% | 68.54% ± 0.10% |

> Results are averaged over 5 folds.

- The training performance decreases compared to Step 1, indicating that dropout effectively regularizes the model.

- While validation loss improves slightly, validation accuracy remains almost unchanged. This suggests that dropout reduces overconfidence without significantly improving classification performance.

- Notably, the standard deviation is significantly reduced compared to previous steps, indicating more stable performance across folds. This may be due to dropout preventing overfitting to specific data splits.

#### step 4. Dropout + Increased Early Stopping Patience

- Add dropout (p=0.5) and increase patience: 5 → 10

##### Results

| Metric | Baseline& Step 1, 2 | Step 3 | Step 4 |
|--------|---------------------|--------|--------|
| Validation Loss | 0.9138 ± 0.0179 | 0.9019 ± 0.0085 | 0.9019 ± 0.0085 |
| Validation Accuracy | 68.59% ± 0.63% | 68.54% ± 0.10% | 68.54% ± 0.10% |

> Results are averaged over 5 folds.

- The results for Step 4 are identical to Step 3, suggesting that patience = 5 may already be sufficient for this setting.

#### step 5. Hidden Layer Addition

- Add convolution layer (32 → 64, kernel_size=3, padding=1, stride=1)
- No dropout

##### Results

| Metric | Baseline& Step 1, 2 | Step 3, 4 | Step 5 |
|--------|---------------------|-----------|--------|
| Validation Loss | 0.9138 ± 0.0179 | 0.9019 ± 0.0085 | 0.8238 ± 0.0208 |
| Validation Accuracy | 68.59% ± 0.63% | 68.54% ± 0.10% | 71.67% ± 0.73% |

> Results are averaged over 5 folds.

- The training accuracy is around 70%, suggesting possible underfitting. To address this, the model capacity was increased by adding a convolutional layer.

- This resulted in the best validation performance among all steps. However, the standard deviation is higher than in previous steps, indicating increased variance.

- This may be due to the higher model capacity, which makes the model more sensitive to data splits.

#### step 6. Pooling Reordering (Conv → Conv → Pool)

- Previous:
  Conv1 → Pool1 → Conv2 → Pool2 → Conv3 → Pool3  

- Updated:
  Conv1 → Conv2 → Pool1 → Conv3 → AdaptiveMaxPool(4×4)

##### Results

| Metric | Baseline& Step 1, 2 | Step 3, 4 | Step 5 | Step 6 |
|--------|---------------------|-----------|--------|--------|
| Validation Loss | 0.9138 ± 0.0179 | 0.9019 ± 0.0085 | 0.8238 ± 0.0208 | 0.7983 ± 0.0237 |
| Validation Accuracy | 68.59% ± 0.63% | 68.54% ± 0.10% | 71.67% ± 0.73% | 72.69% ± 0.79% | 

> Results are averaged over 5 folds.

- The pooling positions were changed to allow more feature extraction at higher resolutions. Adaptive pooling was added to produce a fixed output size of 4×4.

- This resulted in improved validation performance, but the standard deviation also increased.

- This may be due to increased model complexity and reduced early downsampling, which make the model more sensitive to variations across data folds.

#### step 7. Batch Normalization

- Add BatchNorm layers after each convolution layer (Conv → BatchNorm → ReLU)

##### Results

| Metric | Baseline& Step 1, 2 | Step 3, 4 | Step 5 | Step 6 | Step 7 |
|--------|---------------------|-----------|--------|--------|--------|
| Validation Loss | 0.9138 ± 0.0179 | 0.9019 ± 0.0085 | 0.8238 ± 0.0208 | 0.7983 ± 0.0237 | 0.7079 ± 0.0127 |
| Validation Accuracy | 68.59% ± 0.63% | 68.54% ± 0.10% | 71.67% ± 0.73% | 72.69% ± 0.79% | 76.02% ± 0.64% |

> Results are averaged over 5 folds.

- Batch normalization was added after each convolution layer to normalize output distributions.

- This resulted in significantly improved validation performance and a noticeable reduction in standard deviation compared to Step 6.

- This may be because normalization stabilizes training and reduces optimization variance. In addition, its weak regularization effect may further contribute to improved performance and lower variance.

#### step 8. Hidden Layer Addition

- Add convolution layer (64 → 128, kernel_size=3, padding=1, stride=1)

- Updated:
  Conv1 → Conv2 → Pool1 → Conv3 → Conv4 → Pool2 → AdaptiveMaxPool(2×2)

##### Results

| Metric | Baseline& Step 1, 2 | Step 3, 4 | Step 5 | Step 6 | Step 7 | Step 8 |
|--------|---------------------|-----------|--------|--------|--------|--------|
| Validation Loss | 0.9138 ± 0.0179 | 0.9019 ± 0.0085 | 0.8238 ± 0.0208 | 0.7983 ± 0.0237 | 0.7079 ± 0.0127 | 0.6230 ± 0.0121 |
| Validation Accuracy | 68.59% ± 0.63% | 68.54% ± 0.10% | 71.67% ± 0.73% | 72.69% ± 0.79% | 76.02% ± 0.64% | 79.35% ± 0.26% |

> Results are averaged over 5 folds.

- A convolutional layer was added to increase model capacity.

- This resulted in significant improvements in validation performance. Notably, although the model capacity increased, the standard deviation decreased.

- This may be because the increased capacity, combined with batch normalization, enables more stable and consistent feature learning across folds.

---