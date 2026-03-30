# 02_cifar10_cnn

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
- Accuracy: 67.26%

---

### Mini Task

Changes applied:

- Conv layers: 2 → 3
- Kernel size: 3 → 5

Task1 model

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

- Accuracy: 70.78%

Notes: $O = \frac{I + 2P - K}{S} + 1$

- I: Input size
- P: Padding
- K: Kernel size
- S: Stride

---

### Result
The Task 1 model improved accuracy by 3.52% compared to the baseline model.
