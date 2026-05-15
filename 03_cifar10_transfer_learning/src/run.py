from torchvision import models
# from torchinfo import summary
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import argparse
import train as tr

def run(mode, is_aug, opt, scheduler_name):
    aug_transform = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize( mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize( mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ])

    train_datasets = datasets.CIFAR10(
        root = './data',
        train = True,
        download = True,
        transform = aug_transform if is_aug else transform
    )

    test_datasets = datasets.CIFAR10(
        root = './data',
        train = False,
        download = True,
        transform = transform
    )

    train_loader = DataLoader(train_datasets, batch_size = 32, shuffle = True)
    test_loader = DataLoader(test_datasets, batch_size = 32, shuffle = False)

    model = models.resnet18(weights = 'IMAGENET1K_V1')

    if mode == 'freeze':
        for p in model.parameters():
            p.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 10)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    num_epochs = 5


    if mode == 'freeze':
        optimizer = optim.Adam(model.fc.parameters(), lr = 0.001)

    elif mode == 'finetune':
        if opt == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)
        elif opt == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr = 0.0001)
        else:
            raise ValueError("opt must be 'SGD' or 'Adam'")

    else:
        raise ValueError("mode must be 'freeze' or 'finetune'")

    criterion = nn.CrossEntropyLoss()

    tr.train(device, model, train_loader, num_epochs, optimizer, criterion, scheduler_name)
    tr.test(device, model, test_loader, criterion)


if __name__ == '__main__':
    # model = models.resnet18(weights = "IMAGENET1K_V1")
    # #print(model)

    # total_params = sum(p.numel() for p in model.parameters())
    # #print(f"total parameters: {total_params}")

    # summary(model, input_size = (1, 3, 224, 224))

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type = str, default = 'freeze')
    parser.add_argument('--augmentation', action = 'store_true')
    parser.add_argument('--optimizer', type = str, default = 'SGD')
    parser.add_argument('--scheduler', type = str, default = 'StepLR')
    args = parser.parse_args()

    run(args.mode, args.augmentation, args.optimizer, args.scheduler)