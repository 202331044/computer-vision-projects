# 02 CIFAR-10 CNN: Model Performance Improvement

## step 1. Early Stopping

- Add early stopping (Epochs: 10 → 100, patience=5)

### Results

Step 1: No performance change compared to baseline.

> Results are averaged over 5 folds.

- Despite increasing the maximum epochs (10 → 100), early stopping (patience=5) resulted in identical performance, suggesting that the model converged early.

## step 2. Increase Early Stopping Patience

- Increase patience: 5 → 10

### Results

Step 2: No performance change compared to baseline.

> Results are averaged over 5 folds.

- Increasing the early stopping patience (5 → 10) did not affect validation performance.

- However, training accuracy remains significantly higher than validation accuracy, indicating persistent overfitting.

## step 3. Dropout

- Add dropout (p=0.5)
- patience = 5

### Results

| Metric | Baseline | Step 3 |
|--------|----------|--------|
| Validation Loss | 0.9138 ± 0.0179 | 0.9019 ± 0.0085 |
| Validation Accuracy | 68.59% ± 0.63% | 68.54% ± 0.10% |

> Results are averaged over 5 folds.

- The training loss decreases compared to the baseline, indicating that dropout introduces regularization.

- While validation loss improves slightly, validation accuracy remains almost unchanged. This suggests that dropout reduces overconfidence without significantly improving classification performance.

- Notably, the standard deviation is significantly reduced compared to previous steps, indicating more stable performance across folds. This may be due to dropout preventing overfitting to specific data splits.

## step 4. Dropout + Increased Early Stopping Patience

- Add dropout (p=0.5)
- Increase patience: 5 → 10

### Results

Step 4: No performance change compared to step 3.

> Results are averaged over 5 folds.

- The results for Step 4 are identical to Step 3, indicating that increasing the early stopping patience had no effect on performance.

## step 5. Hidden Layer Addition

- Add convolution layer (32 → 64, kernel_size=3, padding=1, stride=1)
- No dropout

### Results

| Metric | Baseline | Step 3 | Step 5 |
|--------|----------|--------|--------|
| Validation Loss | 0.9138 ± 0.0179 | 0.9019 ± 0.0085 | 0.8238 ± 0.0208 |
| Validation Accuracy | 68.59% ± 0.63% | 68.54% ± 0.10% | 71.67% ± 0.73% |

> Results are averaged over 5 folds.

- The training accuracy is around 70%, suggesting possible underfitting. To address this, the model capacity was increased by adding a convolutional layer.

- This resulted in the best validation performance among all steps. However, the standard deviation is higher than in previous steps, indicating increased variance.

- This may be due to the higher model capacity, which makes the model more sensitive to data splits.

## step 6. Pooling Reordering (Conv → Conv → Pool)

- Previous:
  Conv1 → Pool1 → Conv2 → Pool2 → Conv3 → Pool3  

- Updated:
  Conv1 → Conv2 → Pool1 → Conv3 → AdaptiveMaxPool2d((4,4))

### Results

| Metric | Baseline | Step 5 | Step 6 |
|--------|----------|--------|--------|
| Validation Loss | 0.9138 ± 0.0179 | 0.8238 ± 0.0208 | 0.7983 ± 0.0237 |
| Validation Accuracy | 68.59% ± 0.63% | 71.67% ± 0.73% | 72.69% ± 0.79% | 

> Results are averaged over 5 folds.

- The pooling positions were changed to allow more feature extraction at higher resolutions. Adaptive pooling was added to produce a fixed output size of 4×4.

- This resulted in improved validation performance, but the standard deviation also increased.

- This may be due to increased model complexity and reduced early downsampling, which make the model more sensitive to variations across data folds.

## step 7. Batch Normalization

- Add BatchNorm layers after each convolution layer (Conv → BatchNorm → ReLU)

### Results

| Metric | Baseline | Step 6 | Step 7 |
|--------|----------|--------|--------|
| Validation Loss | 0.9138 ± 0.0179 | 0.7983 ± 0.0237 | 0.7079 ± 0.0127 |
| Validation Accuracy | 68.59% ± 0.63% | 72.69% ± 0.79% | 76.02% ± 0.64% |

> Results are averaged over 5 folds.

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

### Results

| Metric | Baseline | Step 7 | Step 8 |
|--------|----------|--------|--------|
| Validation Loss | 0.9138 ± 0.0179 | 0.7079 ± 0.0127 | 0.6230 ± 0.0121 |
| Validation Accuracy | 68.59% ± 0.63% | 76.02% ± 0.64% | 79.35% ± 0.26% |

> Results are averaged over 5 folds.

- A convolutional layer was added to increase model capacity.

- This resulted in significant improvements in validation performance. Notably, although the model capacity increased, the standard deviation decreased.

- This may be because the increased capacity, combined with batch normalization, enables more stable and consistent feature learning across folds.

## Step 9. AdaptiveMaxPool → AdaptiveAvgPool

- Replace AdaptiveMaxPool2d((2,2)) with AdaptiveAvgPool2d((2,2))

### Results

| Metric | Baseline | Step 8 | Step 9 |
|--------|----------|--------|--------|
| Validation Loss | 0.9138 ± 0.0179 | 0.6230 ± 0.0121 | 0.5741 ± 0.0121 |
| Validation Accuracy | 68.59% ± 0.63% | 79.35% ± 0.26% | 80.82% ± 0.47% |

> Results are averaged over 5 folds.

- Step 9 achieved the best performance among all previous steps.

- Average pooling provides a more representative feature abstraction than max pooling, which only selects the maximum activation and may discard useful information.

- The validation loss variance remains similar to Step 8, while the accuracy variance is higher. This is likely due to less extreme feature representations from average pooling, leading to less confident predictions near the decision boundary.

## Step 10. Global Average Pooling (Adaptive)

- Replace AdaptiveAvgPool2d((2,2)) with AdaptiveAvgPool2d((1,1))

- Previous:
  AdaptiveAvgPool2d((2,2)) → Flatten → FC1(512, 128) → FC2(128, 10)

- Updated:
  AdaptiveAvgPool2d((1,1)) → Flatten → FC1(128, 128) → FC2(128, 10)

### Results

| Metric | Baseline | Step 9 | Step 10 |
|--------|----------|--------|---------|
| Validation Loss | 0.9138 ± 0.0179 | 0.5741 ± 0.0121 | 0.6680 ± 0.0482 |
| Validation Accuracy | 68.59% ± 0.63% | 80.82% ± 0.47% | 76.96% ± 2.14% |

> Results are averaged over 5 folds.

- This change resulted in decreased validation performance and increased variance compared to Step 9.

- This can be explained by the loss of spatial information caused by global average pooling, which compresses each feature map into a single value. Furthermore, the reduced dimensionality leads to a smaller fully connected layer, decreasing model capacity and contributing to the performance drop.
