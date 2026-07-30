import torch.nn as nn
import torch
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.d_k = d_k

        self.wq = nn.Linear(d_model, d_k)
        self.wk = nn.Linear(d_model, d_k)
        self.wv = nn.Linear(d_model, d_k)
    
    def forward(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        score = q @ k.transpose(-2, -1)
        score = score / math.sqrt(self.d_k)

        weight = torch.softmax(score, dim = -1)
        output = weight @ v
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def forward(self, x, mask = False):
        batch, seq_len = x.size(0), x.size(1)

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        score = q @ k.transpose(-2, -1)
        score = score / math.sqrt(self.head_dim)

        if mask:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device = x.device), 
            diagonal = 1)
            score = score.masked_fill(causal_mask == 1, float('-inf'))

        weight = torch.softmax(score, dim = -1)

        out = weight @ v

        out = out.transpose(1, 2).contiguous()
        out = out.reshape(batch, seq_len, self.num_heads * self.head_dim)

        out = self.wo(out)

        return out

class _PositionEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        pe = torch.zeros(seq_len, d_model)
        
        for pos in range(seq_len):
            for i in range(d_model):
                if i % 2 == 0:
                    pe[pos, i] = math.sin(pos / pow(10000, i / d_model))
                else:
                    pe[pos, i] = math.cos(pos / pow(10000, i / d_model))

        self.register_buffer('pe', pe)
    
    def forward(self, x):
        
        if x.shape[1:] == self.pe.shape:
            return x + self.pe
        
        raise ValueError('Shape mismatch')


class PositionEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        pe = torch.zeros(max_len, d_model)

        pos = torch.arange(max_len)
        pos = pos.unsqueeze(1)
        dim = torch.arange(0, d_model, 2)
        div_term = torch.pow(10000, dim / d_model)
        pe[:, 0::2] = torch.sin(pos / div_term)
        pe[:, 1::2] = torch.cos(pos / div_term)

        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1)]
        #if x.shape[1:] == self.pe.shape:
        #    return x + self.pe
        
        #raise ValueError('x and pe size mistmatch')
    
    
class EncoderLayer(nn.Module):
    def __init__(self, num_heads, d_model, d_ff):
        super().__init__()

        self.attention = MultiHeadAttention(num_heads, d_model)
        self.dropout = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.ReLU(),
                                  nn.Linear(d_ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        out = self.attention(x)
        out = self.dropout(out)
        out = out + x
        out = self.norm1(out)
        
        out2 = self.ffn(out)
        out2 = self.dropout(out2)
        out2 = out2 + out
        out2 = self.norm2(out2)

        return out2
    
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, N, num_heads, d_ff):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p = 0.1)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = PositionEncoding(max_len, d_model)
        self.layers = nn.ModuleList([ EncoderLayer(num_heads, d_model, d_ff)
                                     for _ in range(N)])
    def forward(self, x):
        out = self.embedding(x) * math.sqrt(self.d_model)
        out = self.position(out)
        out = self.dropout(out)

        for layer in self.layers:
            out = layer(out)
        
        return out

class CrossAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        assert d_model % num_heads == 0 , ('d_model must divided by num_heads')

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
    def forward(self, x, memory):
        batch, target_len = x.shape[:2]
        source_len = memory.size(1)

        q = self.wq(x)
        k = self.wk(memory)
        v = self.wv(memory)

        q = q.reshape(batch, target_len, self.num_heads, self.head_dim)
        k = k.reshape(batch, source_len, self.num_heads, self.head_dim)
        v = v.reshape(batch, source_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        score = q @ k.transpose(-2, -1)
        score = score / math.sqrt(self.head_dim)
        weight = torch.softmax(score, dim = -1)
        out = weight @ v

        out = out.transpose(1, 2).contiguous()

        out = out.reshape(batch, target_len, self.num_heads * self.head_dim)

        out = self.wo(out)

        return out

class DecoderLayer(nn.Module):
    def __init__(self, num_heads, d_model, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(num_heads, d_model)
        self.dropout = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(d_model)

        self.attn2 = CrossAttention(num_heads, d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff),
                                 nn.ReLU(),
                                 nn.Linear(d_ff, d_model))
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, memory):
        out = self.attn(x, mask = True)
        out = self.dropout(out)
        out = self.norm1(out + x)
        
        out2 = self.attn2(out, memory)
        out2 = self.dropout(out2)
        out2 = self.norm2(out2 + out)

        out3 = self.ffn(out2)
        out3 = self.dropout(out3)
        out3 = self.norm3(out3 + out2)

        return out3

class TransformerDecoder(nn.Module):
    def __init__(self, N, vocab_size, d_model, max_len, num_heads, d_ff):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = PositionEncoding(max_len, d_model)
        self.dropout = nn.Dropout(0.1)
        self.layers = nn.ModuleList([DecoderLayer(num_heads, d_model, d_ff) for _ in range(N)])

    def forward(self, x, memory):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.position(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, memory)
        
        return x

class Transformer(nn.Module):
    def __init__(self,
                 N,
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model,
                 src_len,
                 tgt_len,
                 num_heads,
                 d_ff):

        super().__init__()
        self.encoder = TransformerEncoder(src_vocab_size, 
                                          d_model, 
                                          src_len, 
                                          N, 
                                          num_heads, 
                                          d_ff)

        self.decoder = TransformerDecoder(N, 
                                          tgt_vocab_size,
                                          d_model,
                                          tgt_len,
                                          num_heads,
                                          d_ff)
        self.fc = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, src, tgt):
        memory = self.encoder(src)
        out = self.decoder(tgt, memory)
        out = self.fc(out)

        return out