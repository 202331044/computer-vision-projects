# 03 CIFAR-10 Transfer Learning

Transfer learning experiments on CIFAR-10 using a pretrained ResNet-18 and ResNet-50 models.

## 📚 Core Concepts

### Feature Extraction (Freeze)

**Advantages**
- Training is faster because only the FC layer is updated.
- Memory usage is reduced because fewer parameters are updated.
- It works better on small datasets because the risk of overfitting is lower.

**Disadvantages**
- Performance improvement is limited because the model cannot learn new task-specific features.

### Fine-Tuning

**Advantages**
- It can achieve better performance on a new task because the entire model is trainable.

**Disadvantages**
- Training is slower because all model parameters are updated.
- It is more prone to overfitting on small datasets.

### Data Augmentation

- Improve generalization by generating diverse input distributions.
- Reduce overfitting to specific training samples.

### SGD (momentum = 0.9)

- All parameters are updated using the same learning rate.
- Convergence is generally slower than Adam, but generalization performance can sometimes be better.

### Adam

- Parameters are updated using adaptive learning rates based on the 1st and 2nd moment of gradients.
- Convergence is generally faster than SGD.

### StepLR

- The learning rate is reduced by multiplying it by gamma at predefined steps.
- It reduces the learning rate abruptly at specific epochs.
- It is simple and easy to control.
- It often shows stable performance in CNN training.

### CosineAnnealingLR

- The learning rate decreases following a cosine curve.
- T_max determines the length of the cosine cycle.
- The learning rate decreases smoothly compared to StepLR, which can sometimes improve training stability and performance.
- It is often effective in longer training settings.

--- 
## 🚀 Experiment

### CIFAR-10 Dataset

- Image size: 32 × 32 (RGB)
- Classes: 10
- Training data: 50,000 samples
- Test data: 10,000 samples

### ResNet-18 and ResNet-50

- ResNet introduces skip connections (residual connections) to improve gradient flow and enable the training of very deep neural networks.
- Implemented ResNet-18 and ResNet-50 from scratch and verified training on CIFAR-10.
- Used pretrained PyTorch ResNet models for transfer learning experiments and performance comparison.

#### Training Setup

- Pretrained: ImageNet
- Epochs: 5
- Batch size: 32
- Optimizer: Adam
- Learning rate: 1e-4
- Loss function: CrossEntropyLoss

#### Comparison

- Feature Extraction (Freeze) vs Fine-Tuning
- Data Augmentation
- Optimizer (Adam vs SGD)
- Scheduler (None vs StepLR vs CosineAnnealingLR)

#### Results

##### Feature Extraction vs Fine-Tuning

| Method | ResNet18 | ResNet50 |
|--------|-----------|-----------|
| Freeze | 80.75% (0.5754) | 82.71% (0.5071) |
| Fine-Tuning | 94.15% (0.1898) | 94.41% (0.1787) |


##### Data Augmentation

| Augmentation | ResNet18 | ResNet50 |
|--------------|-----------|-----------|
| No Augmentation (10 epochs) | 93.64% (0.2296) | 93.72% (0.2306) |
| RandomCrop + Flip (10 epochs) | 94.32% (0.2027) | 93.88% (0.2149) |


##### Optimizer (Adam vs SGD)

| Optimizer | ResNet18 | ResNet50 |
|-----------|-----------|-----------|
| Adam | 94.15% (0.1898) | 94.41% (0.1787) |
| SGD | 91.78% (0.2772) | 93.58% (0.2071) |


##### Learning Rate Scheduler

| Scheduler | ResNet18 | ResNet50 |
|-----------|-----------|-----------|
| None | 94.15% (0.1898) | 94.41% (0.1787) |
| StepLR | 95.90% (0.1279) | 96.68% (0.1041) |
| CosineAnnealingLR | 95.88% (0.1362) | 96.67% (0.1146) |


👉 [View ResNet-18 Notebook](./experiments/resnet18.ipynb)
👉 [View ResNet-50 Notebook](./experiments/resnet50.ipynb)  
👉 [View Detailed Analysis](./docs/experiment_log.md)

---

### MobileNetV2

- MobileNetV2 uses depthwise separable convolutions (depthwise and pointwise convolutions) to reduce computational cost and the number of parameters.
- It introduces inverted residual blocks with linear bottlenecks to improve efficiency while preserving information.
- Implemented MobileNetV2 from scratch.
- Trained and evaluated the model on CIFAR-10.

#### Training Setup

- Epochs: 5 and 10
- Batch size: 32
- Optimizer: Adam
- Learning rate: 1e-3
- Loss function: CrossEntropyLoss

👉 [View MobileNetV2 Notebook](./experiments/mobilenetv2.ipynb)

---
