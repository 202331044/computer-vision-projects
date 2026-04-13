# 02_cifar10_cnn

## 🧰 Core Libraries

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

- Although the validation loss improves, the validation accuracy slightly decreases, suggesting that the model becomes less overconfident but does not improve classification performance.

#### step 4. Dropout + Increased Early Stopping Patience

- Add dropout (p=0.5) and increase patience: 5 → 10

##### Results

| Metric | Baseline& Step 1, 2 | Step 3 | Step 4 |
|--------|---------------------|--------|--------|
| Validation Loss | 0.9138 ± 0.0179 | 0.9019 ± 0.0085 | 0.9019 ± 0.0085 |
| Validation Accuracy | 68.59% ± 0.63% | 68.54% ± 0.10% | 68.54% ± 0.10% |

> Results are averaged over 5 folds.

- The results for Step 4 are identical to Step 3, suggesting that patience = 5 may already be sufficient for this setting.

---