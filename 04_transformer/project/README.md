# PyTorch Transformer for AG News Classification

Developed a Transformer-based text classification model using PyTorch and evaluated its performance on the AG News dataset.

## 📚 Core Concepts

### Metrics

- **Accuracy**: The proportion of correctly predicted samples among all samples.

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

- **Precision**: The proportion of correctly predicted positive samples among all samples predicted as positive.

$$
Precision = \frac{TP}{TP + FP}
$$

- **Recall**: The proportion of correctly predicted positive samples among all actual positive samples.

$$
Recall = \frac{TP}{TP + FN}
$$

- **F1 Score**: The harmonic mean of Precision and Recall.

$$
F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
$$

- **Confusion Matrix**: A table that summarizes the classification results using TP, TN, FP, and FN.
- **ROC Curve (Receiver Operating Characteristic)**: A curve that plots TPR against FPR at different classification thresholds.
  - **TPR (True Positive Rate)**: The proportion of actual positive samples correctly classified as positive. Equivalent to Recall. (Y-axis)
  - **FPR (False Positive Rate)**: The proportion of actual negative samples incorrectly classified as positive. (X-axis)
- **AUC (Area Under the Curve)**: The area under the ROC curve. A value closer to 1 indicates better classification performance.
    - AUC = 1.0: Perfect classification.
    - AUC = 0.5: Random classification (no discriminative ability).

### Averaging Methods

- **Micro Averaging**: Aggregates TP, FP, and FN across all classes before calculating metrics such as Precision, Recall, and F1 Score.
  - Classes with more samples generally have a greater influence on the overall metrics.
- **Macro Averaging**: Calculates metrics independently for each class and takes their unweighted average.
  - All classes contribute equally, regardless of their sample sizes.

---

## 🚀 Experiments

### AG News Dataset

- Dataset: AG News
- Task: 4-class news topic classification
- Maximum input sequence length: 128

### Training Setup

- Epochs: 10
- `d_model`: 128
- Number of attention heads: 4
- Feed-forward dimension (`d_ff`): 512
- Number of Encoder Layers (`N`): 2
- Dropout: 0.1
- `pad_idx`: 0
- Optimizer: AdamW
- Learning rate: 1e-3
- Loss function: CrossEntropyLoss
- Scheduler: CosineAnnealingLR
- `T_max`: 10

### Results

| Metric | Score |
|--------|------:|
| Loss | 0.3484 |
| Accuracy | 0.9111 |
| Precision (Macro) | 0.9110 |
| Recall (Macro) | 0.9111 |
| F1 Score (Macro) | 0.9109 |
| ROC AUC (OVR, Macro) | 0.9828 |

---

### Model Comparison

#### Encoder Depth Comparison

- Fixed `num_heads` = 4
- `*` indicates the baseline configuration.

| N | Parameters | Accuracy | F1 Score | ROC AUC (OVR, Macro) |
|---:|---:|---:|---:|---:|
| 1 | 3.22M | 80.54% | 0.8048 | 0.9478 |
| * 2 | 3.42M | 91.11% | 0.9109 | 0.9828 |
| 3 | 3.62M | 91.11% | 0.9111 | 0.9830 |

Increasing the number of encoder layers from 1 to 2 significantly improved the overall performance. However, increasing the number of layers from 2 to 3 resulted in almost no performance improvement, while the number of model parameters continued to increase.

#### Attention Heads Comparison

- Fixed `N` = 2
- `*` indicates the baseline configuration.

| num_heads | Parameters | Accuracy | F1 Score | ROC AUC (OVR, Macro) |
|---:|---:|---:|---:|---:|
| 2 | 3.42M | 91.33% | 0.9133 | 0.9832 |
| * 4 | 3.42M | 91.11% | 0.9109 | 0.9828 |
| 8 | 3.42M | 91.32% | 0.9131 | 0.9834 |

Increasing the number of attention heads did not consistently improve performance. The models with `num_heads=2` and `num_heads=8` achieved similar performance, while the number of model parameters remained unchanged.

#### Conclusion

The configuration with `N=2` and `num_heads=2` provides a good balance between model complexity and performance.

👉 [View Notebook](./experiments/run.ipynb)

For result visualizations, see the [`results`](./results/) directory.