# 03 CIFAR-10 Transfer Learning

## 1. Feature Extractor (Freeze) vs Fine-Tuning

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

- When data augmentation was applied, the test accuracy slightly decreased from 94.15% to 93.75%. Although augmentation generally improves generalization performance, the expected improvement was not observed in this experiment.

- This is likely because training for only 5 epochs was insufficient for the model to adapt to the augmented data.

- The experiment was trained for 10 epochs, and the augmentation case achieved the best performance (94.32%) among all settings. This suggests that data augmentation becomes more effective when the model is trained for a sufficient number of epochs.

- In the no-augmentation case, performance slightly decreased compared to 5 epochs (94.15% → 93.64%), indicating that increasing epochs alone does not guarantee improved performance and may lead to minor fluctuations.

- Overall, data augmentation is more effective when combined with sufficient training epochs, while it may show lower performance under short training conditions.

## 3. Optimizer (Adam vs SGD)

### 📊 Performance Comparison

| Optimizer | Loss | Accuracy |
|-----------|------|----------|
| Adam | 0.1898 | 94.15% |
| SGD | 0.2772 | 91.78% |

### 📉 Analysis

The experimental results of Adam and SGD were compared.

- Adam achieved higher training accuracy from the early epochs and converged faster during training. In this experiment, Adam also showed better test accuracy and lower test loss than SGD.

- In contrast, SGD showed slower convergence during the early training stage, resulting in lower performance within 5 epochs.

- These results reflect the tendency of Adam to converge faster than SGD under limited training epochs.

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

- Since the experiment was conducted for only 5 epochs, StepLR was more effective because it reduced the learning rate quickly, leading to faster convergence. In contrast, the advantages of CosineAnnealingLR may not have been fully reflected in such a short training setting.

- Overall, applying a learning rate scheduler improved performance, and StepLR was effective in short training scenarios.

---