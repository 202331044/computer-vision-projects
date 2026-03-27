import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding = 1) #input channel:3, output channel: 16, kernel size: 3x3
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding = 1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32*8*8, 128) 
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x))) # (batch_size, 3, 32, 32) -> conv1: (batch_size, 16, 32, 32) -> pool: (batch_size, 16, 16, 16)
        x = self.pool(self.relu(self.conv2(x))) # (batch_size, 16, 16, 16) -> conv2: (batch_size, 32, 16, 16) -> pool: (batch_size, 32, 8, 8)

        x = x.view(x.size(0), -1) #(batch_size = 32, features = 32*8*8)
        x = self.relu(self.fc1(x)) #(batch_size, features) -> (batch_size, 128)
        x = self.fc2(x) #(batch_size, 128) -> (batch_size, 10)

        return x

def train(epochs, device, train_data, model, loss_function, optimizer):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total = 0
        for images, labels in train_data:
            
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size

        print(f"epoch: {epoch+1} loss: {total_loss/total: .4f}")

def test(device, test_data, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_data:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, prediction = torch.max(outputs, 1)
            correct += (prediction == labels).sum().item()
            total += labels.size(0)

    print(f"Accuracy: {correct/total * 100:.2f}")

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

    train_datasets = torchvision.datasets.CIFAR10(
        root = "./data",
        train = True,
        transform = transform,
        download = True
    )

    test_datasets = torchvision.datasets.CIFAR10(
        root = "./data",
        train = False,
        transform = transform
    )

    train_data = torch.utils.data.DataLoader(train_datasets, batch_size = 32, shuffle = True)
    test_data = torch.utils.data.DataLoader(test_datasets, batch_size = 32, shuffle = False)

    model = CNN().to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    train(3, device, train_data, model, loss_function, optimizer)
    test(device, test_data, model)

if __name__ == "__main__":
    run()