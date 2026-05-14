# 03 CIFAR-10 Transfer Learning

## 1. Feature Extractor (Freeze) vs Fine-Tuning

### 📊 Performance Comparison

| Method | Loss | Accuracy |
|--------|------|----------|
| Freeze | 0.5699 | 80.53% |
| Fine-Tuning | 0.2506 | 92.63% |

### 📉 Analysis

In the freeze setting, only the fully connected (FC) layer is trainable, while in fine-tuning the entire model is trainable.

- Fine-tuning achieved better performance in both loss and accuracy. However, training time increased because the entire model was retrained. 
- The experimental results reflected the characteristics of each method well.

## 2. Data Augmentation

### 📊 Performance Comparison

| Augmentation | Loss | Accuracy |
|--------------|------|----------|
| No Augmentation | 0.1979 | 93.87% |
| RandomCrop + Flip | 0.1680 | 94.74% |

### 📉 Analysis

Data augmentation (RandomCrop and RandomHorizontalFlip) was applied and compared with the baseline setting without augmentation.

- When data augmentation was applied, training performance decreased while test performance improved.
- This experimentally demonstrates that data augmentation improves generalization performance by exposing the model to diverse input variations.

## 3. Optimizer (Adam vs SGD)

### 📊 Performance Comparison

| Optimizer | Loss | Accuracy |
|-----------|------|----------|
| Adam | 0.1994 | 93.91% |
| SGD | 0.2903 | 91.84% |

### 📉 Analysis

The experimental results of Adam and SGD were compared.

- Adam achieved higher training accuracy than SGD from the early epochs and converged faster during training.
- In addition, Adam achieved better test accuracy and lower test loss in this experiment.
- On the other hand, SGD showed relatively slower convergence in the early training stage, but its performance gradually improved as training progressed.
- These results demonstrate the fast convergence characteristic of Adam and the slower but steady convergence behavior of SGD.

## 4. Learning Rate Scheduler

### 📊 Performance Comparison

| Scheduler | Loss | Accuracy |
|-----------|------|----------|
| None | 0.1346 | 95.64% |
| StepLR | 0.1310 | 95.80% |
| CosineAnnealingLR | 0.1453 | 95.58% |

### 📉 Analysis

The experimental results of the baseline setting, StepLR, and CosineAnnealingLR were compared.

- The experimental results showed similar performance among the baseline, StepLR, and CosineAnnealingLR settings.
- Among them, StepLR achieved the highest test accuracy, while CosineAnnealingLR showed slightly lower performance than the baseline in this experiment.
- However, the performance differences were relatively small. This suggests that the effect of learning rate schedulers may not be significant in short training settings such as 5 epochs.
- In addition, CosineAnnealingLR may show its advantages more clearly in longer training settings because it reduces the learning rate more smoothly and gradually.

---