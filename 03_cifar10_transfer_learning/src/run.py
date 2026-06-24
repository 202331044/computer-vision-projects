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

def run(mode, model_name, is_aug, opt, scheduler_name, epochs, lr, is_resize):

    ut.set_seed(42)

    if is_aug:
        aug_transform = ut.get_augmentation()
    
    if mode == 'manual':
        transform = []
        if is_resize:
            transform.append(transforms.Resize((224, 224)))

        transform.extend([ transforms.ToTensor(),
                           transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                std=[0.2470, 0.2435, 0.2616])])
        transform = transforms.Compose(transform)
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
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

    train_loader = DataLoader(train_datasets, batch_size = 32, shuffle = True, 
                              generator = g, num_workers=1, worker_init_fn = ut.seed_worker)

    test_loader = DataLoader(test_datasets, batch_size = 32, shuffle = False)

    model = ut.get_model(model_name)

    if mode == 'freeze':
        for p in model.parameters():
            p.requires_grad = False


    if mode != 'manual':
        #resnet
        #model.fc = nn.Linear(model.fc.in_features, 10)
        
        #mobilenet, efficientnet
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if mode == 'freeze':
        #resnet, manual
        #optimizer = ut.get_optimizer(model.fc.parameters(), opt, lr)
        #mobilenet
        optimizer = ut.get_optimizer(model.classifier.parameters(), opt, lr)
    else:
        optimizer = ut.get_optimizer(model.parameters(), opt, lr)

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

    parser.add_argument('--mode', type = str, default = 'freeze', 
                        choices = ['freeze', 'finetune', 'manual'])
    parser.add_argument('--model', type = str, default = 'resnet18',
                        choices = ['resnet18', 'resnet50', 'resnet18_manual', 
                        'resnet50_manual', 'mobilenetv2_manual',
                        'mobilenetv2', 'efficientnet_manual',
                        'efficientnet'])
    parser.add_argument('--augmentation', action = 'store_true')
    parser.add_argument('--optimizer', type = str, default = 'adam',
                        choices = ['adam', 'sgd'])
    parser.add_argument('--scheduler', type = str, default = 'none',
                        choices = ['none', 'steplr', 'cosinelr'])
    parser.add_argument('--epochs', type = int, default = 5)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--resize', action = 'store_true')

    args = parser.parse_args()

    run(args.mode, args.model, args.augmentation, args.optimizer, 
        args.scheduler, args.epochs, args.lr, args.resize)