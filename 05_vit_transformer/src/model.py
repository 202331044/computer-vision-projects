import torch.nn as nn
import torch
from patch_embedding import PatchEmbedding

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):

        d_model = q.size(-1)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        score = q @ k.transpose(-1, -2) / (d_model ** 0.5)
        weight = torch.softmax(score, dim=-1)

        output = weight @ v

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0, "d_model must be divided by num_heads"

        self.wq = nn.Linear(self.d_model, self.d_model)
        self.wk = nn.Linear(self.d_model, self.d_model)
        self.wv = nn.Linear(self.d_model, self.d_model)
        self.wo = nn.Linear(self.d_model, self.d_model)

    def forward(self, q, k, v):
        head_dim = self.d_model // self.num_heads
        batch_size, tgt_tokens = q.shape[:2]
        src_tokens = k.shape[1]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = q.reshape(batch_size, tgt_tokens, self.num_heads, head_dim)
        k = k.reshape(batch_size, src_tokens, self.num_heads, head_dim)
        v = v.reshape(batch_size, src_tokens, self.num_heads, head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        score = q @ k.transpose(-1, -2) / (head_dim ** 0.5)
        
        weight = torch.softmax(score, dim=-1)
        output = weight @ v

        output = output.transpose(1, 2)
        output = output.reshape(batch_size, tgt_tokens, self.d_model)
        output = self.wo(output)

        return output

class ViTEmbedding(nn.Module):
    def __init__(self, H, W, patch_size, d_model, in_channel):
        super().__init__()
        patch_num = (H // patch_size) * (W // patch_size)
        self.patch_embed = PatchEmbedding(in_channel, patch_size, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, patch_num + 1, d_model))

    def forward(self, x):
        B = x.shape[0]
        output = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        output = torch.concat([cls, output], dim = 1)
        output = output + self.pos_embed

        return output

class MLP(nn.Module):
    def __init__(self, d_model, ratio):
        super().__init__()
        self.fc1= nn.Linear(d_model, d_model * ratio)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(d_model * ratio, d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)

        return x

class ViTEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(0.1)

        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, 4)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.attn(x, x, x)
        x = self.dropout1(x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.dropout2(x)
        x = residual + x

        return x

class ViT(nn.Module):
    def __init__(self, N, H, W, patch_size, num_heads, d_model, class_num, in_channel):
        super().__init__()
        self.embed = ViTEmbedding(H, W, patch_size, d_model, in_channel)
        self.layers = nn.ModuleList([
                                    ViTEncoderBlock(d_model, num_heads)
                                    for _ in range(N)
                                    ])
        self.fc = nn.Linear(d_model, class_num)

    def forward(self, x):
        x = self.embed(x)

        for layer in self.layers:
            x = layer(x)

        x = x[:, 0]
        x = self.fc(x)
        return x