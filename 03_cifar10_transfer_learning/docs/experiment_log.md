# 03 CIFAR-10 Transfer Learning

# ResNet18

## 1. Feature Extractor (Freeze) vs Fine-Tuning

Freeze settings:
- Learning rate: 1e-3

### 📊 Performance Comparison

| Method | Loss | Accuracy |
|--------|------|----------|
| Freeze | 0.5754 | 80.75% |
| Fine-Tuning | 0.1898 | 94.15% |

### 📉 Analysis

In the freeze setting, only the fully connected (FC) layer is trainable, while in fine-tuning the entire model is trainable.

- Fine-tuning achieved better performance in both loss and accuracy. However, training time increased because the entire model was retrained. 

- This result clearly reflects the tendency of each method: the freeze setting enables faster training because only the FC layer is trainable, while fine-tuning achieves better performance because all model parameters are trainable.


## 2. Data Augmentation

### 📊 Performance Comparison

| Augmentation | Loss | Accuracy |
|--------------|------|----------|
| No Augmentation | 0.1898 | 94.15% |
| RandomCrop + Flip | 0.1913 | 93.75% |
| No Augmentation (epochs = 10) | 0.2296 | 93.64% |
| RandomCrop + Flip (epochs = 10) | 0.2027 | 94.32% |

### 📉 Analysis

Data augmentation (RandomCrop and RandomHorizontalFlip) was applied and compared with the baseline setting without augmentation.

- When data augmentation was applied with 5 epochs, the test accuracy slightly decreased from 94.15% to 93.75%. Although data augmentation generally improves generalization performance, the expected improvement was not observed under this setting.

- Since both training and test performance decreased, it appears that 5 epochs were insufficient for the model to properly adapt to the augmented data distribution.

- When the number of epochs was increased to 10, the augmented setting achieved the highest test accuracy (94.32%), indicating that additional training helped the model benefit from augmented samples. Notably, compared with the non-augmented setting at 10 epochs, the augmented model showed lower training performance but higher test accuracy. This suggests that data augmentation reduced overfitting and improved generalization performance.

- However, the loss value of the augmented model at 10 epochs was still higher than that of the 5-epoch baseline without augmentation. Therefore, it is difficult to conclude that augmentation was fully effective under the current training setup, and further experiments with different epoch settings may be necessary.


## 3. Optimizer (Adam vs SGD)

SGD settings:
- Learning rate: 1e-2
- Momentum: 0.9

### 📊 Performance Comparison

| Optimizer | Loss | Accuracy |
|-----------|------|----------|
| Adam | 0.1898 | 94.15% |
| SGD | 0.2772 | 91.78% |

### 📉 Analysis

The experimental results of Adam and SGD were compared.

- In this experiment, Adam achieved higher test accuracy and lower test loss than SGD. In contrast, SGD showed slower convergence during the early stages of training, resulting in lower performance within 5 epochs.

- These results suggest that Adam converges faster than SGD when the number of training epochs is limited.


## 4. Learning Rate Scheduler

### 📊 Performance Comparison

| Scheduler | Loss | Accuracy |
|-----------|------|----------|
| None | 0.1898 | 94.15% |
| StepLR | 0.1279 | 95.90% |
| CosineAnnealingLR | 0.1362 | 95.88% |

### 📉 Analysis

The performances of the baseline model, StepLR, and CosineAnnealingLR were compared.

- StepLR achieved the best performance, while the baseline model without a scheduler showed the lowest performance. StepLR and CosineAnnealingLR produced very similar results.

- Since the experiment was conducted for only 5 epochs, StepLR may have been more effective because it reduced the learning rate quickly, leading to faster convergence. In contrast, the advantages of CosineAnnealingLR may not have been fully reflected in such a short training setting.

- Since the training accuracy was already above 99%, the model may have nearly converged within 5 epochs. Therefore, increasing the number of epochs may not necessarily lead to further performance improvements.

----

# ResNet50

## 1. Feature Extractor (Freeze) vs Fine-Tuning

Freeze settings:
- Learning rate: 1e-3

### 📊 Performance Comparison

| Method | Loss | Accuracy |
|--------|------|----------|
| Freeze | 0.5071 | 82.71% |
| Fine-Tuning | 0.1787 | 94.41% |

### 📉 Analysis

Similar to the ResNet18 experiment, fine-tuning achieved better performance, while the freeze setting required less training time.

- The overall performance ranking was:

  Fine-Tuning (ResNet50) > Fine-Tuning (ResNet18) > Freeze (ResNet50) > Freeze (ResNet18).

- Although Fine-Tuning with ResNet50 achieved the highest accuracy, the performance gap between ResNet50 and ResNet18 was relatively small. However, the training time of ResNet50 was approximately three times longer than that of ResNet18.

- These results suggest that increasing model complexity improved performance only marginally while significantly increasing computational cost.


## 2. Data Augmentation

### 📊 Performance Comparison

| Augmentation | Loss | Accuracy |
|--------------|------|----------|
| No Augmentation (epochs = 10) | 0.2306 | 93.72% |
| RandomCrop + Flip (epochs = 10) | 0.2149 | 93.88% |

### 📉 Analysis

With data augmentation applied, the model achieved slightly better performance than the non-augmented setting under 10 epochs.

- However, compared with the ResNet18 experiment, the augmented ResNet50 model still showed lower performance than the augmented ResNet18 model. 

- The overall performance ranking was:

  ResNet18 + Augmentation > ResNet50 + Augmentation > non-augmented ResNet18 ≈ non-augmented ResNet50.

- One possible explanation is that 10 epochs were still insufficient for ResNet50 to fully benefit from the augmented data, since larger models may require more training iterations to converge properly.

- Another possibility is that ResNet50 may be excessively complex for the CIFAR-10 dataset under the current training setup. However, since the training accuracy remained around 97%, it is difficult to conclude that the model was excessively large or insufficiently trained.


## 3. Optimizer (Adam vs SGD)

SGD settings:
- Learning rate: 1e-2
- Momentum: 0.9

### 📊 Performance Comparison

| Optimizer | Loss | Accuracy |
|-----------|------|----------|
| Adam | 0.1787 | 94.41% |
| SGD | 0.2071 | 93.58% |

### 📉 Analysis

Similar to the ResNet18 experiment, Adam achieved better performance than SGD in the ResNet50 experiment.

- The overall performance ranking was:
  Adam (ResNet50) > Adam (ResNet18) > SGD (ResNet50) > SGD (ResNet18).

- These results suggest that Adam provided faster and more stable optimization under the current training setting, especially when the number of training epochs was limited.


## 4. Learning Rate Scheduler

### 📊 Performance Comparison

| Scheduler | Loss | Accuracy |
|-----------|------|----------|
| None | 0.1787 | 94.41% |
| StepLR | 0.1041 | 96.68% |
| CosineAnnealingLR | 0.1146 | 96.67% |

### 📉 Analysis

Similar to the ResNet18 experiment, the performance ranking was:
  StepLR > CosineAnnealingLR > No Scheduler, indicating that applying a learning rate scheduler effectively improved model performance.

- The overall performance ranking was:

  StepLR (ResNet50) > CosineAnnealingLR (ResNet50) > StepLR (ResNet18) > CosineAnnealingLR (ResNet18).

- These results suggest that learning rate scheduling contributed to more stable optimization and better generalization performance in both ResNet18 and ResNet50 experiments.

---