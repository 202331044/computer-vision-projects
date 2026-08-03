import torch
import model as m
import train as tr
import torch.nn as nn
import torch.optim as optim
import argparse

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

src_vocab = {
    "<PAD>": 0,
    "<BOS>": 1,
    "<EOS>": 2,
    "나는": 3,
    "밥을": 4,
    "먹는다": 5
}

tgt_vocab = {
    "<PAD>": 0,
    "<BOS>": 1,
    "<EOS>": 2,
    "I": 3,
    "eat": 4,
    "rice": 5
}

def run(epochs, lr):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    str_src = ["나는", "밥을", "먹는다", "<EOS>"]
    str_tgt = ["<BOS>", "I", "eat", "rice"]
    str_label = ["I", "eat", "rice", "<EOS>"]

    src_id = [src_vocab[token] for token in str_src]
    tgt_id = [tgt_vocab[token] for token in str_tgt]
    label_id = [tgt_vocab[token] for token in str_label]

    src = torch.tensor(src_id)
    tgt = torch.tensor(tgt_id)
    label = torch.tensor(label_id)

    print("src:", src)
    print("tgt:", tgt)
    print("label:", label)

    #add batch
    src = src.unsqueeze(0)
    tgt = tgt.unsqueeze(0)
    label = label.unsqueeze(0)

    #model
    N = 2
    src_vocab_size = 6
    tgt_vocab_size = 6
    d_model = 16
    src_len = 4
    tgt_len = 4
    num_heads = 4
    d_ff = 64


    transformer = m.Transformer(N,
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model,
                 src_len,
                 tgt_len,
                 num_heads,
                 d_ff)

    seq_len = tgt.size(1)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal = 1).bool()

    transformer = transformer.to(device)
    src = src.to(device)
    tgt = tgt.to(device)
    label = label.to(device)
    causal_mask = causal_mask.to(device)

    optimizer = optim.Adam(transformer.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index = PAD_ID)

    tr.train(epochs, 
          transformer,
          criterion, 
          optimizer, 
          label, 
          src, 
          tgt, 
          tgt_vocab_size, 
          causal_mask,
          PAD_ID)

    max_len = 4
    result = tr.inference(transformer, src, max_len, BOS_ID, EOS_ID)

    id_to_token = { v: k for k, v in tgt_vocab.items()}

    ids = result[0].cpu().tolist()
    tokens = []

    for token_id in ids:
        token = id_to_token[token_id]

        if token == '<EOS>':
            break

        if token not in ['<BOS>', '<PAD>']:
            tokens.append(token)
    
    print()
    print(str_src)
    print(" ".join(tokens))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default = 1)
    parser.add_argument("--lr", type = float, default = 1e-3)
    args = parser.parse_args()
    run(args.epochs, args.lr)