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

`train_test_split`: split and shuffle data, index return

--- 
## CIFAR10 Experiment

### CIFAR10 Dataset

- Image size: 32 × 32 (RGB)
- Classes: 10
- Training data: 50,000 samples
- Test data: 10,000 samples

### Model Architecture

Baseline Model

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

- Training setup
    - Epochs: 3
    - Batch size: 32

- Train Loss: 0.8974
- Train Accuracy: 68.33%
- Validation Loss: 0.9125
- Validation Accuracy: 67.28%

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

    - Train Loss: 0.8639
    - Train Accuracy: 69.65%
    - Validation Loss: 0.9045
    - Validation Accuracy: 67.90%

2. Epoch Increase (3 → 10)

    The model was trained for 10 epochs to improve learning.

    - Train Loss: 0.3256
    - Train Accuracy: 88.35%
    - Validation Loss: 0.9534
    - Validation Accuracy: 72.66%

    **Analysis**

    - The validation accuracy is lower than the training accuracy, suggesting that the model may be overfitting to the training data.

---