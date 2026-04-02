# 02_cifar10_cnn

## Key Components

`sklearn`: Data splitting, preprocessing, and evaluation utilities

`sklearn.model_selection`: Train/validation splitting and cross-validation module

---

## Core Concepts

Notes: $O = \frac{I + 2P - K}{S} + 1$

- I: Input size
- P: Padding
- K: Kernel size
- S: Stride

`train_test_split`: splits and returns data

`StratifiedKFold().split()` returns train and validation indices, ensuring that class distribution is preserved
--- 
## CIFAR10 Experiment

### CIFAR10 Dataset

- Image size: 32 × 32 (RGB)
- Classes: 10
- Training data: 50,000 samples
- Test data: 10,000 samples

### Model Architecture

Baseline Model

- Training setup
    - Epochs: 3
    - Batch size: 32

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

** Best Model (selected by validation accuracy) **

| Metric | Value |
|--------|------|
| Epoch | 3 |
| Train Loss | 0.8974 |
| Train Accuracy | 68.33% |
| Validation Loss | 0.9125 |
| Validation Accuracy | 67.28% |

---

### Mini Task: Model Performance Improvement

1. Conv layers: 2 → 3, Kernel size: 3 → 5

    Improved feature extraction by increasing model depth and kernel size.

    **Task1 model**

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

    ** Best Model (selected by validation accuracy) **

    | Metric | Value |
    |--------|------|
    | Epoch | 3 |
    | Train Loss | 0.8639 |
    | Train Accuracy | 69.65% |
    | Validation Loss | 0.9045 |
    | Validation Accuracy | 67.90% |

2. Epoch: 3 → 10

    The model was trained for 10 epochs to improve learning.

    **Best Model (selected by validation accuracy)**

    | Metric | Value |
    |--------|------|
    | Epoch | 6 |
    | Train Loss | 0.5484 |
    | Train Accuracy | 80.72% |
    | Validation Loss | 0.8244 |
    | Validation Accuracy | 72.78% |

    **Analysis**

    - The validation accuracy is lower than the training accuracy, suggesting that the model may be overfitting to the training data.

3. Training Improvements

    Several training strategies were applied to improve model performance:

    - Early Stopping: epochs = 100, patience = 7
    - Batch Size: 32 → 64
    - Optimizer Comparison:
        - Adam
        - SGD
        - AdamW
        - SGD + Momentum
    
    **Best Model (AdamW, selected by validation accuracy)**

    | Metric | Value |
    |--------|------|
    | Epoch | 9 |
    | Train Loss | 0.4213 |
    | Train Accuracy | 85.02% |
    | Validation Loss | 0.8645 |
    | Validation Accuracy | 73.72% |

4. Add Cross-Validation

    Implemented stratified k-fold cross-validation to evaluate model performance on different data subsets.

    - Optimizer: AdamW
    - K-fold: 5

    | Metric | Value |
    |--------|------|
    | Validation Mean Loss | 1.0119|
    | Validation Mean Accuracy | 68.91% |

    **Analysis**

    - Since the average accuracy was used, the result is lower than the previous attempt (3.), 
    which used the highest accuracy.
---