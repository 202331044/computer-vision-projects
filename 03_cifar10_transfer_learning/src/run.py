from torchvision import models
# from torchinfo import summary
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import argparse
import time

def run(mode):
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
        transform = transform
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

    if(mode == 'freeze'):
        for p in model.parameters():
            p.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 10)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    num_epochs = 5


    if(mode == 'freeze'):
        optimizer = optim.Adam(model.fc.parameters(), lr = 0.001)
    elif(mode == 'finetune'):
        optimizer = optim.Adam(model.parameters(), lr = 0.0001)
    else:
        raise ValueError("mode must be 'freeze' or 'finetune'")

    criterion = nn.CrossEntropyLoss()

    train(device, model, train_loader, num_epochs, optimizer, criterion)
    test(device, model, test_loader, criterion)


def train(device, model, train_loader, num_epochs, optimizer, criterion):
    for epoch in range(num_epochs):
        start_time = time.time()

        model.train()

        running_loss = 0
        total = 0
        correct = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            batch_size = labels.size(0)
            total += batch_size

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()

            _, predictions = torch.max(outputs, 1)
            correct += (predictions == labels).sum().item()

            running_loss += loss.item() * batch_size

        elapsed_time = time.time() - start_time

        accuracy = correct / total * 100
        print(f'[Train {epoch + 1} / {num_epochs}]',
              f'Loss: {running_loss/total:.4f}, Accuracy: {accuracy:.2f}%, ',
              f'Time: {elapsed_time:.2f}s')


def test(device, model, test_loader, criterion):
    model.eval()

    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            batch_size = labels.size(0)
            total += batch_size

            outputs = model(images)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item() * batch_size

            _, predictions = torch.max(outputs, 1)
            correct += (predictions == labels).sum().item()

    accuracy = correct / total * 100

    print(f'[Test] Loss: {running_loss/total:.4f}, Accuracy: {accuracy:.2f}%')
      
      
if __name__ == '__main__':
    # model = models.resnet18(weights = "IMAGENET1K_V1")
    # #print(model)

    # total_params = sum(p.numel() for p in model.parameters())
    # #print(f"total parameters: {total_params}")

    # summary(model, input_size = (1, 3, 224, 224))

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type = str, default = 'freeze')
    args = parser.parse_args()

    run(args.mode)
