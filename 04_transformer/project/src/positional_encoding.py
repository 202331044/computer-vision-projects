import torch.nn as nn
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        #div_term = (torch.arange(0, d_model, 2) / d_model)
        pe = torch.zeros(max_len, d_model)
        #pe[:, 0::2] = torch.sin(position / torch.pow(10000, div_term))
        #pe[:, 1::2] = torch.cos(position / torch.pow(10000, div_term))

        term = torch.exp(torch.arange(0, d_model, 2).float() / d_model * 
                             -torch.log(torch.tensor(10000.0)))
        pe[:, 0::2] = torch.sin(position * term)
        pe[:, 1::2] = torch.cos(position * term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]
