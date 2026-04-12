# 01_mnist_mlp

## 🧰 Core Libraries

### torch

`torch`: PyTorch
- Core framework for deep learning
- Supports tensor operations (CPU, GPU), automatic differentiation, and neural network construction and training

`torch.no_grad()`: Context manager that disables gradient computation

`torch.max()`: Function that returns the maximum value or its index

`torch.device`: Object that specifies where computations will be performed (CPU or CUDA GPU)

`torch.utils.data.DataLoader`: Class that loads data from a dataset, creates mini-batches, and makes it easier to train models

`torch.manual_seed()`: Function that fixes randomness in PyTorch (CPU operations, weight initialization, dropout, tensor operations)

`torch.cuda.manual_seed()`: Function that fixes randomness for the current GPU

`torch.cuda.manual_seed_all()`: Function that fixes randomness across multiple GPUs

`torch.backends.cudnn.deterministic`: Flag that forces deterministic GPU operations for reproducibility

`torch.backends.cudnn.benchmark`: Flag that disables GPU optimization for reproducible results

`torch.nn`: Module that provides tools for creating neural network layers and models

`torch.nn.functional`: Module that provides functional APIs for neural network operations

`torch.nn.Module`: Base class for building all neural network models

`nn.CrossEntropyLoss()`: Loss function that internally combines LogSoftmax and NLLLoss

`nn.Dropout(p)`: Layer that randomly deactivates neurons during training to prevent overfitting


### torchvision

`torchvision`: Library specialized for computer vision
- Provides popular datasets and pre-trained models
- Supports image processing functionalities

`torchvision.transforms`: Module for image preprocessing and data augmentation

`torchvision.transforms.ToTensor()`: Transform that converts images from (H, W, C) to (C, H, W) and scales pixel values from 0–255 to 0.0–1.0 


### argparse

`argparse`: Python standard library used for handling command-line execution options

`argparse.ArgumentParser`: Class that creates a parser to handle command-line options


### random

`random.seed()`: Function that fixes Python-level randomness (e.g., shuffle, random.sample)

### numpy

`np.random.seed()`: Function that fixes NumPy randomness (array shuffling and random number generation)


### Training & Evaluation

`model.train()`: Method that sets the model to training mode  
- Activates dropout  
- Uses batch statistics in batch normalization  

`model.eval()`: Method that sets the model to evaluation mode  
- Deactivates dropout  
- Uses learned mean and variance in batch normalization  


### Tensor Operations

`.item()`: Method that converts a scalar tensor to a Python number 

---

## 📚 Core Concepts

Tensor

- Multi-dimensional array used to store data in ML/DL

- All PyTorch models accept tensors as input

Dropout

- Regularization method that randomly deactivates neurons during training to prevent overfitting

- Used only during training (disabled during validation and testing)

- Typically applied after ReLU in MLPs, and on a per-layer basis

- In CNNs, applied less frequently after convolution layers and more often after fully connected layers

Model Comparison

- Only one variable is changed at a time for fair comparison
- Random seed is fixed for each fold in cross-validation

Pickle

- Method for saving Python objects to a file (serialization)

---

## 🚀 Experiment

### MNIST Dataset
- Image size: 28 × 28 (grayscale)
- Classes: 10 (handwritten digits 0–9)
- Training data: 60,000 samples
- Test data: 10,000 samples

### Model Architecture

#### Baseline Model

| Layer       | Output Shape     | Details                     |
|-------------|------------------|-----------------------------|
| Input       | (28, 28)         |                             |
| Flatten     | (784)            |                             |
| FC1         | (128)            | ReLU                        |
| FC2         | (10)             |                             |

### Training setup
- Cross-validation: Stratified K-Fold (K=5)
- Epochs: 10
- Batch size: 32
- Optimizer: Adam

### Results

| Metric | Value |
|--------|------|
| Validation Loss | 0.1088 |
| Validation Accuracy | 96.86% |

> Results are averaged over 5 folds.

---

### Mini Task: Model Performance Improvement

#### Step 1. Hidden Layer Size Improvement

- Hidden layer size: 128 → 256

##### Results

| Metric | Baseline | Step 1 |
|--------|----------------|----------------|
| Validation Loss | 0.1088 | 0.0997 |
| Validation Accuracy | 96.86% | 97.17% |

> Results are averaged over 5 folds.

> Improved accuracy by +0.31% and reduced validation loss.

> This suggests that increasing model capacity helps the model capture more complex patterns in the data.

#### Step 2. Training Improvements

- Add early stopping (Epochs: 10 → 100, patience=5)

| Metric | Baseline | Step 1 | Step 2 |
|--------|----------|--------|--------|
| Validation Loss | 0.1088 | 0.0997 | 0.0996 |
| Validation Accuracy | 96.86% | 97.17% | 97.24% |

> Results are averaged over 5 folds.

> Slight improvement over Step 1 (+0.07% accuracy) with a slight reduction in loss.

> Early stopping had limited impact, suggesting the model was not significantly overfitting.


#### Step 3. Drop Out

- Add dropout (p=0.3)

| Metric | Baseline | Step 1 | Step 2 | Step 3 |
|--------|----------|--------|--------|--------|
| Validation Loss | 0.1088 | 0.0997 | 0.0996 | 0.0906 |
| Validation Accuracy | 96.86% | 97.17% | 97.24% | 97.44% |

> Results are averaged over 5 folds.

> Improved performance over Step 2 (+0.20% accuracy) with a notable reduction in validation loss.

> Early stopping had limited impact, suggesting that the model converged early, while dropout provided stronger regularization.

---
