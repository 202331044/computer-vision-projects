from dataset import AGNewsDataset, collate_fn
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
import torch
from model import TextTransformer
from train import train_model
from datasets import load_dataset
import torch.nn as nn
import argparse
from pathlib import Path
import visualize as vis
from evaluate import evaluate

def run(epochs, nheads, nlayers):

    ag_news_train = load_dataset('fancyzhx/ag_news', split='train')
    ag_news_test = load_dataset('fancyzhx/ag_news', split='test')

    split = ag_news_train.train_test_split(test_size=0.1,
                              seed=42)

    train_datasets = AGNewsDataset(split['train'])
    val_datasets = AGNewsDataset(split['test'])  
    test_datasets = AGNewsDataset(ag_news_test)

    tk = Tokenizer(train_datasets, min_freq=10)
    
    train_data = []
    val_data = []
    test_data = []

    for i in range(len(train_datasets)):
        text, label = train_datasets[i]
        train_data.append([tk.encode(text), label])

    for i in range(len(val_datasets)):
        text, label = val_datasets[i]
        val_data.append([tk.encode(text), label])

    for i in range(len(test_datasets)):
        text, label = test_datasets[i]
        test_data.append([tk.encode(text), label])

    train_loader = DataLoader(train_data, 
                              batch_size=32, 
                              collate_fn=collate_fn)

    val_loader = DataLoader(val_data, 
                            batch_size=32, 
                            collate_fn=collate_fn)

    test_loader = DataLoader(test_data,
                             batch_size=32,
                             collate_fn=collate_fn)

    vocab_size = len(tk.idx2word)
    d_model = 128
    pad_idx = 0
    max_len = 128
    num_heads = nheads
    d_ff = 512
    N = nlayers
    num_classes = 4
    dropout = 0.1

    model = TextTransformer(vocab_size=vocab_size,
                            d_model=d_model,
                            pad_idx=pad_idx,
                            max_len=max_len,
                            num_heads=num_heads,
                            d_ff=d_ff,
                            N=N,
                            num_classes=num_classes,
                            dropout=dropout)
    


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    checkpoint_dir = Path("../checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    #checkpoint_path = checkpoint_dir / "best_model_baseline.pt"
    checkpoint_path = checkpoint_dir / "best_model_cmp.pt"
    
    config = {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "max_len": max_len,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "num_layers": N,
        "dropout": dropout
    }

    history = train_model(epochs, 
                model, 
                device, 
                train_loader, 
                val_loader, 
                optimizer, 
                criterion,
                scheduler,
                config,
                save_path=checkpoint_path)
                
    # fig_dir = Path("../results")
    # fig_dir.mkdir(exist_ok=True)
    
    # loss_path = fig_dir / "train_val_loss.png"
    # acc_path = fig_dir / "train_val_acc.png"

    # vis.plot_loss(history, loss_path)
    # vis.plot_acc(history, acc_path)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    results = evaluate(model, test_loader, device, criterion)

    print("Evaluation Results")
    print("num_heads: ", num_heads)
    print("N: ", N)
    print("Parameters: ", sum(p.numel() for p in checkpoint['model_state_dict'].values()))
    # print(f"Loss: {results['loss']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    # print(f"Precision: {results['precision']:.4f}")
    # print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"ROC AUC (OVR, Macro): {results['macro_roc_auc']:.4f}")
    
    # class_names = ['World', 'Sports', 'Business', 'Sci_Tech']
    # roc_auc_dir = Path("../results")
    # roc_auc_dir.mkdir(exist_ok=True)
    # vis.plot_roc_auc_curve(results['fpr'], results['tpr'], 
    #                    class_names, results['roc_auc'], roc_auc_dir)

    # cm_dir = Path("../results")
    # cm_dir.mkdir(exist_ok=True)
    # cm_path = cm_dir/"normalized_confusion_matrix.png"
    # vis.plot_confusion_matrix(results['confusion_matrix'], class_names, cm_path)
    
if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--nheads', type=int, default=4)
    parser.add_argument('--nlayers', type=int, default=2)
    args = parser.parse_args()
    run(args.epochs, args.nheads, args.nlayers)