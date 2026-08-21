import torch.nn as nn
import positional_encoding as pe

class TextTransformer(nn.Module):
    def __init__(self, 
                vocab_size,
                d_model,
                pad_idx,
                max_len,
                num_heads,
                d_ff,
                N,
                num_classes,
                dropout):

        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=d_model,
                                      padding_idx=self.pad_idx)

        self.posEncoding = pe.PositionalEncoding(max_len, d_model)

        layer = nn.TransformerEncoderLayer(d_model=d_model,
                                           nhead=num_heads,
                                           dim_feedforward=d_ff,
                                           dropout=dropout,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer,
                                             num_layers=N)
        
        self.fc = nn.Linear(d_model, num_classes)


    def forward(self, x):

        padding_mask = x == self.pad_idx

        x = self.embedding(x) * (self.embedding.embedding_dim ** 0.5)
        x = self.posEncoding(x)
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        x = x[:, 0, :]
        logits = self.fc(x)

        return logits