# 04 Transformer

Study the Transformer architecture by implementing it from scratch and using PyTorch implementations.

## 📚 Core Concepts

### RNN (Recurrent Neural Network)

A recurrent neural network that processes sequential data using a hidden state to store contextual information.

**Problems**
- **Sequential Dependency**: Tokens must be processed sequentially, making parallelization difficult.
- **Long-Term Dependency**: Older information can be gradually lost as it is passed through hidden states.

---

### LSTM (Long Short-Term Memory)

An improved RNN architecture that uses a cell state and gates to better preserve long-term information.

- **Forget Gate**: Determines which information to remove from the cell state.
- **Input Gate**: Determines which new information to add to the cell state.
- **Output Gate**: Determines which information to output as the hidden state.
- **Cell State**: Carries long-term information through the sequence.

**Advantages**
- Mitigates the long-term dependency problem compared to RNN.
- Can preserve important information for longer sequences.

**Disadvantages**
- Still requires sequential processing.
- Sequential dependency makes parallelization difficult.

---

### Seq2Seq

An Encoder-Decoder architecture commonly implemented using RNN or LSTM.

- **Encoder**: Converts the input sequence into a contextual representation.
- **Decoder**: Generates the output sequence using the contextual representation and previously generated tokens.

**Problem**
- A single context vector must contain the information needed to generate the entire output sequence, causing an **information bottleneck**.
- Still suffers from the sequential dependency of RNN/LSTM.

---

### Attention

Allows the decoder to selectively focus on different parts of the input sequence when generating each output token.

**Scaled Dot-Product Attention**

$$
\text{Attention}(Q,K,V)=
\text{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

- **Query (Q)**: Represents what information the current token is looking for.
- **Key (K)**: Represents what information each input token contains for matching.
- **Value (V)**: Contains the actual information to be aggregated.
- **Attention Score**: Measures the similarity between Query and Key.
- **Attention Weight**: Obtained by applying softmax to the attention scores.

**Advantages**
- Generates a new context vector for each output step.
- Reduces the information bottleneck of Seq2Seq.

**Disadvantages**
- When combined with RNN/LSTM, sequential dependency still remains.

---

## 🤖 Transformer

A sequence model based on attention mechanisms, without recurrence or convolution.

Each token can attend to other tokens in the sequence, while masked self-attention in the decoder prevents attending to future tokens.

**Advantages**
- Removes sequential dependency.
- Enables efficient parallel processing on GPUs.
- Allows each token to directly interact with other tokens in the sequence.

### Architecture

```text
Input
  │
  ▼
Token Embedding
  │
  ▼
Scaling (× √d_model)
  │
  ▼
Positional Encoding
  │
  ▼
┌──────────────────────────────┐
│          Encoder × N         │
│                              │
│  Self-Attention              │
│  Feed-Forward Network        │
│  Residual & LayerNorm        │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│          Decoder × N         │
│                              │
│  Masked Self-Attention       │
│  Cross-Attention             │
│  Feed-Forward Network        │
│  Residual & LayerNorm        │
└──────────────┬───────────────┘
               │
               ▼
          Linear + Softmax
               │
               ▼
             Output
```

#### Token Embedding

Converts token IDs into `d_model`-dimensional vectors.

The embedding is multiplied by √d_model to increase its magnitude before positional encoding is added.

#### Positional Encoding

Self-Attention does not inherently contain information about token order, so positional information is added to the embeddings.

$$
PE_{(pos,2i)}=
\sin
\left(
\frac{pos}{10000^{2i/d_{model}}}
\right)
$$

$$
PE_{(pos,2i+1)}=
\cos
\left(
\frac{pos}{10000^{2i/d_{model}}}
\right)
$$

**Characteristics**
- Does not require learnable parameters.
- Uses sinusoidal functions with different frequencies to represent positions.
- Provides positional information to the Transformer.

---

#### Encoder

Each Encoder layer consists of:

```text
                 ┌─────────────────────────┐
                 │ Layer Normalization     │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Multi-Head              │
                 │ Self-Attention          │
                 └────────────┬────────────┘
                              │
                              ▼
                           Dropout
                              │
                              ▼
Input ─────────────────────── ⊕
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Layer Normalization     │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Feed-Forward Network    │
                 └────────────┬────────────┘
                              │
                              ▼
                           Dropout
                              │
                              ▼
Self-Attention Output ─────── ⊕
                              │
                              ▼
                            Output
```

##### Multi-Head Self-Attention

Splits the attention mechanism into multiple heads so that different types of relationships between tokens can be learned simultaneously.

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

Each head computes scaled dot-product attention, and the results are concatenated and projected.

##### Feed-Forward Network

Applies a fully connected network independently to each token representation.

$$
FFN(x)=W_2\text{ReLU}(W_1x+b_1)+b_2
$$

---

#### Decoder

Each Decoder layer consists of:

```text
                          ┌─────────────────────────┐
                          │ Layer Normalization     │
                          └────────────┬────────────┘
                                       │
                                       ▼
                          ┌─────────────────────────┐
                          │ Masked Multi-Head       │
                          │ Self-Attention          │
                          └────────────┬────────────┘
                                       │
                                       ▼
                                    Dropout
                                       │
                                       ▼
Input ──────────────────────────────── ⊕
                                       │
                                       ▼
                          ┌─────────────────────────┐
                          │ Layer Normalization     │
                          └────────────┬────────────┘
                                       │
                                       ▼
                          ┌─────────────────────────┐
                          │ Cross-Attention         │
                          └────────────┬────────────┘
                                       │
                                       ▼
                                    Dropout
                                       │
                                       ▼
Masked Self-Attention Output ──────── ⊕
                                       │
                                       ▼
                          ┌─────────────────────────┐
                          │ Layer Normalization     │
                          └────────────┬────────────┘
                                       │
                                       ▼
                          ┌─────────────────────────┐
                          │ Feed-Forward Network    │
                          └────────────┬────────────┘
                                       │
                                       ▼
                                    Dropout
                                       │
                                       ▼
Cross-Attention Output ────────────── ⊕
                                       │
                                       ▼
                                     Output
```

##### Masked Multi-Head Self-Attention

Prevents the decoder from attending to future tokens that have not yet been generated.

**Purpose**
- Prevents information leakage from future tokens.
- Ensures that each position can only attend to itself and previously generated tokens.

#### Encoder-Decoder Attention (Cross-Attention)

Connects the Decoder to the Encoder output.

- **Query**: Decoder representation
- **Key**: Encoder output
- **Value**: Encoder output

This allows the Decoder to selectively retrieve relevant information from the Encoder output.

---

## 🚀 Experiments
### Implementation from scratch

#### Training Setup

- Epochs: 5
- Optimizer: Adam
- Learning rate: 1e-2
- Loss function: CrossEntropyLoss
- Number of Encoder/Decoder Layers (`N`): 2
- Source vocabulary size: 6
- Target vocabulary size: 6
- `d_model`: 16
- Source sequence length: 4
- Target sequence length: 4
- Number of attention heads: 4
- Feed-forward dimension (`d_ff`): 64

#### Results

- Verified that the Transformer implemented from scratch can correctly translate the given toy input sentence.

👉 [View Notebook](./experiments/run.ipynb)

---

### PyTorch Transformer

- Implemented a Transformer-based text classification model using PyTorch.
- Evaluated the model on the AG News dataset.

#### AG News Dataset

- Classes: 4
- Maximum input sequence length: 128

#### Training Setup

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

#### Results

| Metric | Score |
|--------|------:|
| Loss | 0.3342 |
| Accuracy | 91.00% |
| Precision (Macro) | 0.9100 |
| Recall (Macro) | 0.9099 |
| F1 Score (Macro) | 0.9098 |
| ROC AUC (OVR, Macro) | 0.9830 |

👉 [View Notebook](./project/experiments/run.ipynb)

---