# 02 CIFAR-10 CNN: Model Performance Improvement

## step 1. Early Stopping

- Add early stopping (Epochs: 10 → 100, patience=5)

### 📊 Performance Comparison

Step 1: No performance change compared to baseline.

> Results are averaged over 5 folds.

### 📉 Analysis

- Despite increasing the maximum epochs (10 → 100), early stopping (patience=5) resulted in identical performance, suggesting that the model converged early.


## step 2. Increase Early Stopping Patience

- Increase patience: 5 → 10

### 📊 Performance Comparison

Step 2: No performance change compared to baseline.

> Results are averaged over 5 folds.

### 📉 Analysis

- Increasing the early stopping patience (5 → 10) did not affect validation performance.

- However, training accuracy remains significantly higher than validation accuracy, indicating persistent overfitting.

## step 3. Dropout

- Add dropout (p=0.5)
- patience = 5

### 📊 Performance Comparison

| Metric | Baseline | Step 3 |
|--------|----------|--------|
| Validation Loss | 0.9138 ± 0.0179 | 0.9019 ± 0.0085 |
| Validation Accuracy | 68.59% ± 0.63% | 68.54% ± 0.10% |

> Results are averaged over 5 folds.

### 📉 Analysis

- The training loss decreases compared to the baseline, indicating that dropout introduces regularization.

- While validation loss improves slightly, validation accuracy remains almost unchanged. This suggests that dropout reduces overconfidence without significantly improving classification performance.

- Notably, the standard deviation is significantly reduced compared to previous steps, indicating more stable performance across folds. This may be due to dropout preventing overfitting to specific data splits.

## step 4. Dropout + Increased Early Stopping Patience

- Add dropout (p=0.5)
- Increase patience: 5 → 10

### 📊 Performance Comparison

Step 4: No performance change compared to step 3.

> Results are averaged over 5 folds.

### 📉 Analysis

- The results for Step 4 are identical to Step 3, indicating that increasing the early stopping patience had no effect on performance.

## step 5. Hidden Layer Addition

- Add convolution layer (32 → 64, kernel_size=3, padding=1, stride=1)
- No dropout

### 📊 Performance Comparison

| Metric | Baseline | Step 3 | Step 5 |
|--------|----------|--------|--------|
| Validation Loss | 0.9138 ± 0.0179 | 0.9019 ± 0.0085 | 0.8238 ± 0.0208 |
| Validation Accuracy | 68.59% ± 0.63% | 68.54% ± 0.10% | 71.67% ± 0.73% |

> Results are averaged over 5 folds.

### 📉 Analysis

- The training accuracy is around 70%, suggesting possible underfitting. To address this, the model capacity was increased by adding a convolutional layer.

- This resulted in the best validation performance among all steps. However, the standard deviation is higher than in previous steps, indicating increased variance.

- This may be due to the higher model capacity, which makes the model more sensitive to data splits.

## step 6. Pooling Reordering (Conv → Conv → Pool)

- Previous:
  Conv1 → Pool1 → Conv2 → Pool2 → Conv3 → Pool3  

- Updated:
  Conv1 → Conv2 → Pool1 → Conv3 → AdaptiveMaxPool2d((4,4))

### 📊 Performance Comparison

| Metric | Baseline | Step 5 | Step 6 |
|--------|----------|--------|--------|
| Validation Loss | 0.9138 ± 0.0179 | 0.8238 ± 0.0208 | 0.7983 ± 0.0237 |
| Validation Accuracy | 68.59% ± 0.63% | 71.67% ± 0.73% | 72.69% ± 0.79% | 

> Results are averaged over 5 folds.

### 📉 Analysis

- The pooling positions were changed to allow more feature extraction at higher resolutions. Adaptive pooling was added to produce a fixed output size of 4×4.

- This resulted in improved validation performance, but the standard deviation also increased.

- This may be due to increased model complexity and reduced early downsampling, which make the model more sensitive to variations across data folds.

## step 7. Batch Normalization

- Add BatchNorm layers after each convolution layer (Conv → BatchNorm → ReLU)

### 📊 Performance Comparison

| Metric | Baseline | Step 6 | Step 7 |
|--------|----------|--------|--------|
| Validation Loss | 0.9138 ± 0.0179 | 0.7983 ± 0.0237 | 0.7079 ± 0.0127 |
| Validation Accuracy | 68.59% ± 0.63% | 72.69% ± 0.79% | 76.02% ± 0.64% |

> Results are averaged over 5 folds.

### 📉 Analysis

- Batch normalization was added after each convolution layer to normalize output distributions.

- This resulted in significantly improved validation performance and a noticeable reduction in standard deviation compared to Step 6.

- This may be because normalization stabilizes training and improves optimization.

## step 8. Combined Architecture Update (Multiple Changes)

- Add convolution layer (64 → 128, kernel_size=3, padding=1, stride=1)
- Add pooling layer (Pool2, MaxPool2d)
- Change AdaptiveMaxPool output size: (4×4) → (2×2)

- Previous:
  Conv1 → Conv2 → Pool1 → Conv3 → AdaptiveMaxPool2d((4,4))

- Updated:
  Conv1 → Conv2 → Pool1 → Conv3 → Conv4 → Pool2 → AdaptiveMaxPool2d((2,2))

### 📊 Performance Comparison

| Metric | Baseline | Step 7 | Step 8 |
|--------|----------|--------|--------|
| Validation Loss | 0.9138 ± 0.0179 | 0.7079 ± 0.0127 | 0.6230 ± 0.0121 |
| Validation Accuracy | 68.59% ± 0.63% | 76.02% ± 0.64% | 79.35% ± 0.26% |

> Results are averaged over 5 folds.

### 📉 Analysis

- A convolutional layer was added to increase model capacity.

- This resulted in significant improvements in validation performance. Notably, although the model capacity increased, the standard deviation decreased.

- This may be because the increased capacity, combined with batch normalization, enables more stable and consistent feature learning across folds.

## Step 9. AdaptiveMaxPool → AdaptiveAvgPool

- Replace AdaptiveMaxPool2d((2,2)) with AdaptiveAvgPool2d((2,2))

### 📊 Performance Comparison

| Metric | Baseline | Step 8 | Step 9 |
|--------|----------|--------|--------|
| Validation Loss | 0.9138 ± 0.0179 | 0.6230 ± 0.0121 | 0.5741 ± 0.0121 |
| Validation Accuracy | 68.59% ± 0.63% | 79.35% ± 0.26% | 80.82% ± 0.47% |

> Results are averaged over 5 folds.

### 📉 Analysis

- Step 9 achieved the best performance among all previous steps.

- Average pooling provides a more representative feature abstraction than max pooling, which only selects the maximum activation and may discard useful information.

- The validation loss variance remains similar to Step 8, while the accuracy variance is higher. This is likely due to less extreme feature representations from average pooling, leading to less confident predictions near the decision boundary.

## Step 10. Global Average Pooling (Adaptive)

- Replace AdaptiveAvgPool2d((2,2)) with AdaptiveAvgPool2d((1,1))

- Previous:
  AdaptiveAvgPool2d((2,2)) → Flatten → FC1(512, 128) → FC2(128, 10)

- Updated:
  AdaptiveAvgPool2d((1,1)) → Flatten → FC1(128, 128) → FC2(128, 10)

### 📊 Performance Comparison

| Metric | Baseline | Step 9 | Step 10 |
|--------|----------|--------|---------|
| Validation Loss | 0.9138 ± 0.0179 | 0.5741 ± 0.0121 | 0.6680 ± 0.0482 |
| Validation Accuracy | 68.59% ± 0.63% | 80.82% ± 0.47% | 76.96% ± 2.14% |

> Results are averaged over 5 folds.

### 📉 Analysis

- This change resulted in decreased validation performance and increased variance compared to Step 9.

- This can be explained by the loss of spatial information caused by global average pooling, which compresses each feature map into a single value. Furthermore, the reduced dimensionality leads to a smaller fully connected layer, decreasing model capacity and contributing to the performance drop.

## Step 11. Residual Block: With vs Without Skip Connection

### 📊 Performance Comparison

| Metric | Step 9 | Step 10 | Step 11 (No Skip) | Step 11 (Skip) |
|--------|--------|---------|-------------------|----------------|
| Validation Loss | 0.5741 ± 0.0121 | 0.6680 ± 0.0482 | 0.5489 ± 0.0139 | 0.5396 ± 0.0216 |
| Validation Accuracy | 80.82% ± 0.47% | 76.96% ± 2.14% | 82.00% ± 0.88% | 82.86% ± 0.97% |

> Results are averaged over 5 folds.

### 📉 Analysis

- Applying a residual block resulted in better performance in both validation loss and accuracy compared to the previous best model (Step 9). However, the standard deviation increased, indicating reduced stability.

- Notably, even without the skip connection, the residual block achieved better performance than Step 9. This suggests that increased network depth may have improved model capacity and feature extraction capability.

- When the skip connection is enabled, the input is directly added to the output, which facilitates gradient flow and makes optimization easier. This leads to more efficient convergence and the best overall performance.

- The increased standard deviation is mainly due to higher model capacity, which increases sensitivity to initialization and data variation. Additionally, improved optimization flexibility may lead to convergence to different local minima, increasing variability across runs.

### Step 12. Residual Block: No Conv Bias

### 📊 Performance Comparison

| Metric | Step 11 (No Skip) | Step 11 (Skip) | Step 12 (No Skip) | Step 12 (Skip) |
|--------|-------------------|----------------|-------------------|----------------|
| Validation Loss | 0.5489 ± 0.0139 | 0.5396 ± 0.0216 | 0.5371 ± 0.0135 | 0.5341 ± 0.0217 |
| Validation Accuracy | 82.00% ± 0.88% | 82.86% ± 0.97% | 82.49% ± 0.76% | 81.94% ± 0.93% |

> Results are averaged over 5 folds.

### 📉 Analysis

- The bias term of the convolution layers within the residual block was removed. When Batch Normalization is applied, the bias becomes largely redundant, as its effect is absorbed by the shift parameter (β) of the BatchNorm layer.

- As a result, the overall performance remained almost unchanged. The validation loss slightly decreased, while the accuracy showed minor variations (increasing in the no-skip case and decreasing in the skip case). The standard deviation also remained similar, indicating no significant change in stability.

- These small differences may be due to changes in training dynamics. Specifically, removing the bias can slightly alter the feature distribution before Batch Normalization, leading to minor variations in performance.

### Step 13. Stride-Based Downsampling without Pooling

### 📊 Performance Comparison

| Metric | Step 12 (No Skip) | Step 12 (Skip) | Step 13 (No Skip) | Step 13 (Skip) |
|--------|-------------------|----------------|-------------------|----------------|
| Validation Loss | 0.5371 ± 0.0135 | 0.5341 ± 0.0217 | 0.6519 ± 0.0083 | 0.6833 ± 0.0169 |
| Validation Accuracy | 82.49% ± 0.76% | 81.94% ± 0.93% | 77.86% ± 0.60% | 76.75% ± 1.02% |

> Results are averaged over 5 folds.

### 📉 Analysis

- Max pooling was replaced with stride-based downsampling to enable learnable feature compression, but this led to lower performance than Step 12 in both skip and no-skip settings.

- Although both approaches reduce spatial resolution, they differ in how information is preserved: max pooling retains strong local activations through a fixed operation, introducing a strong inductive bias, whereas stride-based downsampling relies on learned filters, offering greater flexibility but not guaranteeing preservation of strong features.

- Interestingly, Step 13 exhibits reduced variance across folds (lower standard deviation), likely due to more uniform feature aggregation in stride-based convolution, which reduces reliance on highly activated local responses. However, this comes at the cost of reduced average performance, suggesting more stable but less expressive representations.

### Step 14. Double Residual Blocks per Stage

- Previous:
[Residual Block ×2] → Pool → [Residual Block ×2] → Pool

- Updated:
[Residual Block ×4] → Pool → [Residual Block ×4] → Pool

### 📊 Performance Comparison

| Metric | Step 12 (No Skip) | Step 12 (Skip) | Step 14 (No Skip) | Step 14 (Skip) |
|--------|-------------------|----------------|-------------------|----------------|
| Validation Loss | 0.5371 ± 0.0135 | 0.5341 ± 0.0217 | 0.5918 ± 0.0355 | 0.5107 ± 0.0130 |
| Validation Accuracy | 82.49% ± 0.76% | 81.94% ± 0.93% | 80.20% ± 1.44% | 83.57% ± 0.77% |

> Results are averaged over 5 folds.

### 📉 Analysis

- In Step 12, skip connections did not improve performance, suggesting the network was too shallow to benefit from them.

- In Step 14, increasing the depth by stacking more residual blocks led to the best performance with skip connections. The deeper network enables more expressive high-level feature extraction. Skip connections further enhance this by preserving input information through identity mappings, improving gradient flow and preventing feature degradation, allowing both low- and high-level features to be effectively utilized.

- In terms of stability, the model without skip connections showed increased standard deviation (e.g., accuracy std 1.44%) as depth increased, whereas the model with skip connections maintained a low std (0.77%), similar to Step 12.

- Overall, deeper networks improve performance through richer feature extraction, while skip connections amplify this effect and ensure stable training.

### Step 15. Data Augmentation

The following data augmentation techniques were applied to the Step 14 model training pipeline:

- **Random Cropping with Padding (32×32, padding=4)**  
  → improves robustness to spatial translations

- **Random Horizontal Flipping (p=0.5)**  
  → improves left-right invariance

### 📊 Performance Comparison

| Metric | Step 14 (Skip) | Step 15 |
|--------|----------------|---------|
| Validation Loss (Best by Loss) | 0.5107 ± 0.0130 | **0.3829 ± 0.0038** |
| Validation Accuracy (Best by Loss) | 83.57% ± 0.77% | **88.09% ± 0.37%** |
| Validation Loss (Best by Accuracy) | - | 0.3977 ± 0.0205 |
| Validation Accuracy (Best by Accuracy) | - | 88.11% ± 0.38% |
| Generalization Gap (Best Loss-based) | _ | **4.48%** |

> Results are averaged over 5 folds.

### 📉 Analysis

- Loss and accuracy metrics at the best-accuracy point were additionally recorded for a more detailed evaluation of model performance. 

- Training performance at the best-loss point was recorded to better analyze overfitting behavior.

- Step 15 achieves the best overall performance among all tested models. It also shows reduced standard deviation compared to Step 14, indicating improved stability across folds. This improvement is attributed to the increased feature diversity and generalization effect introduced by data augmentation.

- The gap between training and validation accuracy at the best-loss point is below 5%, suggesting low overfitting.

---