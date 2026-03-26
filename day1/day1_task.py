import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        flat_x = x.view(-1, 28*28)
        x1 = F.relu(self.fc1(flat_x))
        x2 = self.fc2(x1)
        return x2

def train(device, data, model, loss_function, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total = 0
        for images, labels in data:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)

            batch_size = labels.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size
            total += batch_size

        print(f"epoch: {epoch + 1} loss: {total_loss/total: .4f}")

def test(device, data, model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, prediction = torch.max(outputs, 1)
            
            correct += (prediction == labels).sum().item()
            total += labels.size(0)

    print(f"Accuracy: {correct/total * 100 : .2f}%")

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()
    train_datasets = torchvision.datasets.MNIST(
        root = "./data",
        train = True,
        transform = transform,
        download = True
    )

    test_datasets = torchvision.datasets.MNIST(
        root = "./data",
        train = False,
        transform = transform
    )

    train_data = torch.utils.data.DataLoader(train_datasets, batch_size = 128, shuffle = True)
    test_data = torch.utils.data.DataLoader(test_datasets, batch_size = 128, shuffle = False)

    model = SimpleNN().to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    train(device, train_data, model, loss_function, optimizer, 5)
    test(device, test_data, model)

if __name__ == "__main__":
    run()