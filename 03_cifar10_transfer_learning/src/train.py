import time
import torch.optim as optim
import torch

def train(device, model, train_loader, num_epochs, optimizer, criterion, scheduler_name):
    
    scheduler = None
    
    if scheduler_name == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 2, gamma = 0.1)
    elif scheduler_name == 'CosineLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = num_epochs)

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

        if scheduler is not None:
            scheduler.step()

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