# 01_mnist_mlp

## Core Libraries

### torch

`torch`: PyTorch
- Core framework for deep learning
- Supports tensor operations (CPU, GPU), automatic differentiation, and neural network construction and training

`torch.no_grad()`: Context manager for disabling gradient computation  

`torch.max()`: Function for returning the maximum value or the index of the maximum value  

`torch.device`: Object that specifies where computations will be performed (CPU or CUDA GPU)

`torch.utils.data.DataLoader`: Class that loads data from a dataset, creates mini-batches, and makes it easier to train models

`torch.nn`: Module for creating neural network layers and models

`torch.nn.functional`: Functional API module

`torch.nn.Module`: Class for building all neural network models  
- All PyTorch models must inherit from this class  

`nn.CrossEntropyLoss()`: Loss function that internally combines LogSoftmax and NLLLoss


### torchvision

`torchvision`: Library specialized for computer vision
- Provides popular datasets and pre-trained models
- Supports image processing functionalities

`torchvision.transforms`: Module for image preprocessing and data augmentation

`torchvision.transforms.ToTensor()`: Transform that converts images from (H, W, C) to (C, H, W) and scales pixel values from 0–255 to 0.0–1.0 


### argparse

`argparse`: Python standard library
- Used for handling command-line execution options

`argparse.ArgumentParser`: Class that creates a parser to handle command-line options


### Training & Evaluation

`model.train()`: Method for setting the model to training mode  
- Activates dropout  
- Uses batch statistics in batch normalization  

`model.eval()`: Method for setting the model to evaluation mode  
- Deactivates dropout  
- Uses learned mean and variance in batch normalization  


### Tensor Operations

`.item()`: Function for converting a scalar tensor to a Python number  

---

## Core Concepts

Tensor

- A multi-dimensional array used to store data in ML/DL

- All PyTorch models accept tensors as input

---

## Experiment

### MNIST Dataset
- Image size: 28 × 28 (grayscale)
- Classes: 10 (handwritten digits 0–9)
- Training data: 60,000 samples
- Test data: 10,000 samples

### Model Architecture

**Baseline Model**

- Training setup
    - Epochs: 3
    - Batch size: 64

| Layer       | Output Shape     | Details                     |
|-------------|------------------|-----------------------------|
| Input       | (28, 28)         |                             |
| Flatten     | (784)            |                             |
| FC1         | (128)            | ReLU                        |
| FC2         | (10)             |                             |


- Accuracy: 96.68%

### Mini Task

**Changes applied:**

- Epochs: 3 → 5
- Batch size: 64 → 128
- Hidden layer size: 128 → 256

**Task1 model**

- Accuracy: 97.46%

**Analysis**

The Task 1 model improved accuracy by 0.78% compared to the baseline model.