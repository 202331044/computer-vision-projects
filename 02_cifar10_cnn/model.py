import torch.nn as nn

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

class Task1CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, padding = 2) #output size = (input size + 2p - kernel size)/stride + 1 
        self.conv2 = nn.Conv2d(16, 32, 5, padding = 2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding = 2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*4*4, 128) 
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x))) 
        x = self.pool(self.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x