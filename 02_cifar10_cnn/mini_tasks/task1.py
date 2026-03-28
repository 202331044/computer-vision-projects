# ##Mini task
# - Conv layers 2 -> 3
# - Kernel size 3 -> 5

import torch.nn as nn

class CNN(nn.Module):
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