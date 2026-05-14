# 03 CIFAR-10 Transfer Learning

Transfer learning experiments on CIFAR-10 using a pretrained ResNet-18 model.

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

### ResNet-18

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

| Method | Loss | Accuracy |
|--------|------|----------|
| Freeze | 0.5699 | 80.53% |
| Fine-Tuning | 0.2506 | 92.63% |

---

##### Data Augmentation

| Augmentation | Loss | Accuracy |
|--------------|------|----------|
| No Augmentation | 0.1979 | 93.87% |
| RandomCrop + Flip | 0.1680 | 94.74% |

---

##### Optimizer Comparison

| Optimizer | Loss | Accuracy |
|-----------|------|----------|
| SGD | 0.2903 | 91.84% |
| Adam | 0.1994 | 93.91% |

---

##### Scheduler Comparison

| Scheduler | Loss | Accuracy |
|-----------|------|----------|
| None | 0.1346 | 95.64% |
| StepLR | 0.1310 | 95.80% |
| CosineAnnealingLR | 0.1453 | 95.58% |

👉 [View Detailed Results](./experiments/resnet18.ipynb)

---
