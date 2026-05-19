from torchvision import models
# from torchinfo import summary
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import argparse
import train as tr
import model as md
import utils as ut

def run(mode, model_name, is_aug, aug_type, opt, scheduler_name, epochs):

    ut.set_seed(42)

    if is_aug:
        aug_transform = ut.get_augmentation(aug_type)
        
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

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(train_datasets, batch_size = 32, shuffle = True, generator = g, num_workers=1,
    worker_init_fn = ut.seed_worker)

    test_loader = DataLoader(test_datasets, batch_size = 32, shuffle = False)

    model = ut.get_model(model_name)

    if mode == 'freeze':
        for p in model.parameters():
            p.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 10)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = ut.get_optimizer(mode, model, optimizer)

    criterion = nn.CrossEntropyLoss()

    tr.train(device, model, train_loader, epochs, optimizer, criterion, scheduler_name)
    tr.test(device, model, test_loader, criterion)


if __name__ == '__main__':
    # model = models.resnet18(weights = "IMAGENET1K_V1")
    # #print(model)

    # total_params = sum(p.numel() for p in model.parameters())
    # #print(f"total parameters: {total_params}")

    # summary(model, input_size = (1, 3, 224, 224))

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type = str, default = 'freeze')
    parser.add_argument('--model', type = str, default = 'ResNet18')
    parser.add_argument('--augmentation', action = 'store_true')
    parser.add_argument('--optimizer', type = str, default = 'Adam')
    parser.add_argument('--scheduler', type = str, default = 'None')
    parser.add_argument('--epochs', type = int, default = 5)
    parser.add_argument('--aug-type', type = str, default = 'base')

    args = parser.parse_args()

    run(args.mode, args.model, args.augmentation, args.aug_type, args.optimizer, args.scheduler, args.epochs)