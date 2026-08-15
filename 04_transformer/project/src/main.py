from dataset import AGNewsDataset, collate_fn
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
import torch
from model import TextTransformer
from train import train_model
from datasets import load_dataset
import torch.nn as nn

def run():

    ag_news = load_dataset('fancyzhx/ag_news', split='train')
    split = ag_news.train_test_split(test_size=0.1,
                              seed=42)
    train_datasets = AGNewsDataset(split['train'])
    val_datasets = AGNewsDataset(split['test'])  
    tk = Tokenizer(train_datasets, min_freq=10)
    
    train_data = []
    val_data = []

    for i in range(len(train_datasets)):
        text, label = train_datasets[i]
        train_data.append([tk.encode(text), label])

    for i in range(len(val_datasets)):
        text, label = val_datasets[i]
        val_data.append([tk.encode(text), label])


    train_loader = DataLoader(train_data, 
                              batch_size = 32, 
                              collate_fn = collate_fn)

    val_loader = DataLoader(val_data, 
                              batch_size = 32, 
                              collate_fn = collate_fn)

    model = TextTransformer(vocab_size=len(tk.idx2word),
                            d_model=128,
                            pad_idx=0,
                            max_len=128,
                            num_heads=4,
                            d_ff=512,
                            N=2,
                            num_classes=4)
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    epochs = 10
    train_model(epochs, 
                model, 
                device, 
                train_loader, 
                val_loader, 
                optimizer, 
                criterion,
                scheduler)

if __name__ == '__main__' :
    run()