# 01_mnist_mlp

## Key PyTorch Modules

`torch`: PyTorch core library

`torch.nn`: Neural network layers and model components

`torch.nn.functional`: Functional operations

`torchvision`: Datasets and image-related utilities

`torchvision.transforms`: Image preprocessing tools

---

## Core Concepts

Tensor
- A multi-dimensional array used to store data in ML/DL
- All PyTorch models accept tensors as input
- transforms.ToTensor(): Converts images from (H, W, C) to (C, H, W) and scales pixel values from 0–255 to 0.0–1.0

`nn.Module`: All PyTorch models must inherit from torch.nn.Module.

`nn.CrossEntropyLoss`: Internally applies log-softmax + Negative Log Likelihood Loss (NLLLoss)

`model.train()`: Sets the model to training mode (enables Dropout and updates BatchNorm statistics)

`model.eval()`: Sets the model to evaluation mode (disables Dropout and uses fixed BatchNorm statistics)

`torch.no_grad()`: Disables gradient computation → used during inference to save memory and speed up computation

`.item()`: Converts a single-value tensor into a Python float

`torch.max()`: Returns: max value, index (predicted class)

---

## MNIST Experiment

### MNIST Dataset
- Image size: 28 × 28 (grayscale)
- Classes: 10 (handwritten digits 0–9)
- Training data: 60,000 samples
- Test data: 10,000 samples

### Model Architecture

Baseline Model
- Fully connected (2 layers)
- Input: 28×28 → 128
- Output: 10 classes
- Activation: ReLU
- Training setup
    - Epochs: 3
    - Batch size: 64
- Accuracy: 96.68%

### Mini Task

Changes applied:

- Epochs: 3 → 5
- Batch size: 64 → 128
- Hidden layer size: 128 → 256

Task1 model
- Fully connected (2 layers)
- Input: 28×28 → 256
- Output: 10 classes
- Activation: ReLU
- Training setup
    - Epochs: 5
    - Batch size: 128
- Accuracy: 97.46%

### Result
The Task 1 model improved accuracy by 0.78% compared to the baseline model.
