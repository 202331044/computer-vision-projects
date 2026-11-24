# 05. Vision Transformer (ViT)

Implemented a Vision Transformer (ViT) model using PyTorch and evaluated its performance on the CIFAR-10 image classification dataset.

## 🤖 Core Concepts

### Vision Transformer (ViT)

- A Transformer-based model for image classification.
- Splits images into patches and converts them into embedding vectors.
- Adds positional embeddings and a learnable CLS token to the patch sequence.
- Uses Transformer Encoder layers to learn image representations for classification.

### Key Differences from NLP Transformers

- **Patch Embedding**: Images are divided into patches, which are projected into embedding vectors and treated as tokens.
- **Positional Embedding**: Learnable positional embeddings encode the spatial positions of image patches.
- **CLS Token**: A learnable classification token is prepended to the patch sequence to aggregate image-level information.
- **Transformer Encoder**: Processes patch embeddings using self-attention, similar to NLP Transformers.

---

## 🚀 Experiments

### CIFAR-10 Dataset

- Dataset: CIFAR-10
- Task: 10-class image classification
- Image Size: 32 × 32
- Number of Channels: 3 (RGB)

### Model Configuration
- Patch size: 4 × 4
- Model dimension (`d_model`): 128
- Number of attention heads: 4
- Feed-forward dimension (`d_ff`): 512
- Number of Encoder Layers: 4
- Dropout: 0.1

### Training Setup
- Epochs: 20
- Optimizer: AdamW
- Learning rate: 1e-3
- Loss function: CrossEntropyLoss

### Results

| Metric | Score |
|--------|------:|
| Test Loss | 1.1020 |
| Test Accuracy | 62.11% |

👉 [View Notebook](./experiments/run.ipynb)