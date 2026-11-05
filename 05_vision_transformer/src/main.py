import torch
import model as md
from patch_embedding import PatchEmbedding
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import torch.nn as nn 
import argparse 
from torch.utils.data import random_split
from train import train_model, evaluate
from pathlib import Path

def run(epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean = [0.5, 0.5, 0.5],
                                                     std = [0.5, 0.5, 0.5])
                                 ])
    train_datasets = datasets.CIFAR10(root='./data',
                    train=True,
                    download=True,
                    transform=transform)
    #small_datasets = Subset(train_datasets, range(100))
    #train_loader = DataLoader(small_datasets, batch_size=16, shuffle=True)

    train_size = int(len(train_datasets) * 0.8)
    val_size = len(train_datasets) - train_size

    train_dataset, val_dataset = random_split(train_datasets, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    N = 4
    in_channel, H, W = train_datasets[0][0].shape
    patch_size = 4
    num_heads = 4
    class_num = 10
    d_model = 128
    model = md.ViT(N, H, W, patch_size, num_heads, d_model, class_num, in_channel)   

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)

    checkpoint_dir = Path("./checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.pt"

    history = train_model(model, 
                train_loader, 
                device, 
                criterion, 
                optimizer, 
                epochs, 
                val_loader,
                save_path=checkpoint_path)

    test_datasets = datasets.CIFAR10(root='./data',
                    train=False,
                    download=True,
                    transform=transform)

    test_loader = DataLoader(test_datasets, batch_size=32, shuffle=False)
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = evaluate(model, 
                                   test_loader, 
                                   device, 
                                   criterion)
    
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    # N = 4
    # H = 32
    # W = 32
    # patch_size = 4
    # num_heads = 4
    # d_model = 128
    # class_num = 10
    # in_channel = 3

    # m = model.ViT(
    # N=N,
    # H=H,
    # W=W,
    # patch_size=patch_size,
    # num_heads=num_heads,
    # d_model=d_model,
    # class_num=class_num,
    # in_channel=in_channel
    # )

    # img = torch.randn(8, in_channel, H, W)
    # output = m(img)
    # print(f"ViT:{output.shape}")

    # pe = PatchEmbedding(in_channel, patch_size, d_model)
    # output = pe(img)
    # print(f"patchEmbedding:{output.shape}")

    # ve = model.ViTEmbedding(H, W, patch_size, d_model, in_channel)
    # output = ve(img)
    # print(f"ViTEmbedding:{output.shape}")

    # attn = model.MultiHeadAttention(d_model, num_heads)
    # img2 = ve(img)
    # output = attn(img2, img2, img2)
    # print(f"MultiHeadAttention:{output.shape}")

    # mlp = model.MLP(d_model, ratio=4)
    # img3 = attn(img2, img2, img2)
    # output = mlp(img3)
    # print(f"MLP:{output.shape}")

    # encoder = model.ViTEncoderBlock(d_model, num_heads)
    # output = encoder(img2)
    # print(f"ViTEncoderBlock:{output.shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    run(args.epochs)