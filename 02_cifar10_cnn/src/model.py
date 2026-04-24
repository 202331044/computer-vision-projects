import torch.nn as nn
import torch

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


class Residual(nn.Module):
    def __init__(self, in_ch, out_ch, stride = 1, use_residual=False):
        super().__init__()

        self.use_residual = use_residual

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride = stride, padding = 1, bias = False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding = 1, bias = False)

        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU()
        self.skip = nn.Identity()

        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride = stride, padding = 0, bias = False),
                nn.BatchNorm2d(out_ch)
            )
    
    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.use_residual:
            out = out + identity
        
        out = self.relu(out)

        return out

class Task1CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding = 1) #output size = (input size + 2p - kernel size)/stride + 1 
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding = 1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32*8*8, 128) 
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x))) 
        x = self.pool(self.relu(self.conv2(x))) 

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class Task2CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding = 1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding = 1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding = 1)

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

class Task3CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding = 1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding = 1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding = 1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((4,4))
        
        self.fc1 = nn.Linear(64*4*4, 128) 
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = self.relu(self.conv3(x))
        x = self.adaptive_pool(x)

        #x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class Task4CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding = 1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding = 1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding = 1)
        
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((4,4))

        self.fc1 = nn.Linear(64*4*4, 128) 
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)

        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class Task5CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding = 1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding = 1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding = 1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding = 1)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((2,2))

        self.fc1 = nn.Linear(128*2*2, 128) 
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        x = self.adaptive_pool(x)

        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class Task6CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding = 1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding = 1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding = 1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding = 1)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2,2))

        self.fc1 = nn.Linear(128*2*2, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class Task7CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding = 1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding = 1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding = 1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding = 1)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class Task8CNN(nn.Module):
    def __init__(self, use_residual=False):
        super().__init__()
        self.block1 = Residual(3, 16, use_residual = use_residual)
        self.block2 = Residual(16, 32, use_residual = use_residual)
        self.block3 = Residual(32, 64, use_residual = use_residual)
        self.block4 = Residual(64, 128, use_residual = use_residual)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)

        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class Task9CNN(nn.Module):
    def __init__(self, use_residual=False):
        super().__init__()
        self.block1 = Residual(3, 16, 1, use_residual)
        self.block2 = Residual(16, 32, 2, use_residual)
        self.block3 = Residual(32, 64, 2, use_residual)
        self.block4 = Residual(64, 128, 2, use_residual)

        self.relu = nn.ReLU()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        x = self.block3(x)
        x = self.block4(x)

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x