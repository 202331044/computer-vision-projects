import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace = True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out

class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.layer1 = self._make_layer(64, 3, stride = 1)
        self.layer2 = self._make_layer(128, 4, stride = 2)
        self.layer3 = self._make_layer(256, 6, stride = 2)
        self.layer4 = self._make_layer(512, 3, stride = 2)
        
        self.adaptivepool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, 1000)
    
    def _make_layer(self, planes, blocks, stride = 1):
        layers = []

        downsample = None

        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential( nn.Conv2d(self.inplanes, planes * 4, stride = stride, kernel_size = 1, bias = False),
                                        nn.BatchNorm2d(planes * 4))
        
        layers.append(Bottleneck(self.inplanes, planes, stride = stride, downsample = downsample))

        self.inplanes = planes * 4

        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pool1(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.adaptivepool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

class ResNet50CIFAR(nn.Module):
    def __init__(self):
        super().__init__()

        self.inplane = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 3, 1)
        self.layer2 = self._make_layer(128, 4, 2)
        self.layer3 = self._make_layer(256, 6, 2)
        self.layer4 = self._make_layer(512, 3, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, 10)

    def _make_layer(self, plane, block, stride):
        layers = []
        downsample = None

        if stride != 1 or self.inplane != plane * 4:
            downsample = nn.Sequential(nn.Conv2d(self.inplane, plane * 4, kernel_size= 1, stride = stride, bias = False),
                                       nn.BatchNorm2d(plane * 4))
        
        layers.append(Bottleneck(self.inplane, plane, stride= stride, downsample=downsample))

        self.inplane = plane * 4
        
        for _ in range(1, block):
            layers.append(Bottleneck(self.inplane, plane))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

class Residual(nn.Module):
    def __init__(self, inplane, outplane, stride, downsample = None):
        super().__init__()

        self.conv1 = nn.Conv2d(inplane, outplane, kernel_size = 3, stride = stride, padding = 1, bias = False) 
        self.bn1 = nn.BatchNorm2d(outplane)
        self.relu = nn.ReLU(inplace = True)

        self.conv2 = nn.Conv2d(outplane, outplane, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(outplane)

        self.downsample = None

        if downsample is not None:
            self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity

        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.pool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.layer1 = self._make_layer(64, 64, stride = 1)
        
        self.layer2  = self._make_layer(64, 128, stride = 2)
        
        self.layer3 = self._make_layer(128, 256, stride = 2)
        
        self.layer4 = self._make_layer(256, 512, stride = 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, 1000)
    
    def _make_layer(self, inplane, outplane, stride):

        layers = []
        downsample = None

        if stride != 1 or inplane != outplane:
            downsample = nn.Sequential(nn.Conv2d(inplane, outplane, kernel_size = 1, stride = stride, bias = False),
                                       nn.BatchNorm2d(outplane))
            
        layers.append(Residual(inplane, outplane, stride = stride, downsample = downsample))
        layers.append(Residual(outplane, outplane, stride = 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.pool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
    
class ResNet18CIFAR(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 64, stride = 1)
        self.layer2 = self._make_layer(64, 128, stride = 2)
        self.layer3 = self._make_layer(128, 256, stride = 2)
        self.layer4 = self._make_layer(256, 512, stride = 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, 10)
    
    def _make_layer(self, inplane, outplane, stride):
        layers = []
        downsample = None

        if stride != 1 or inplane != outplane:
            downsample = nn.Sequential(nn.Conv2d(inplane, outplane, kernel_size = 1, stride= stride, bias = False),
                                       nn.BatchNorm2d(outplane))
            
        layers.append(Residual(inplane, outplane, stride, downsample))
        layers.append(Residual(outplane, outplane, 1))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


class Bottleneck2(nn.Module):
    def __init__(self, input, output, t, s):
        super().__init__()

        self.expand = (t != 1)
        self.use_residual = (input == output and s == 1)
        
        if self.expand:
            self.conv1 = nn.Conv2d(input, input * t,
                                   kernel_size = 1,
                                   stride = 1,
                                   bias = False)
            self.bn1 = nn.BatchNorm2d(input * t)
            

        self.relu = nn.ReLU6(inplace = True)
        self.hidden_dim = input * t if self.expand else input

        self.conv2 = nn.Conv2d(self.hidden_dim,
                               self.hidden_dim,
                               kernel_size = 3,
                               stride = s, 
                               padding = 1,
                               groups = self.hidden_dim,
                               bias = False)

        self.bn2 = nn.BatchNorm2d(self.hidden_dim)

        self.conv3 = nn.Conv2d(self.hidden_dim, output, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(output)

    def forward(self, x):

        identity = x

        if self.expand:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)

        if self.use_residual:
            x += identity

        return x

class MobileNetV2(nn.Module):
    def __init__(self, num_classes = 10, cifar = False):
        super().__init__()

        first_stride = 1 if cifar else 2
        #first_stride = 2

        self.conv1 = nn.Conv2d(3, 32, 
                               kernel_size = 3, 
                               stride = first_stride,
                               padding = 1,
                               bias = False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU6(inplace = True)
        
        self.layer1 = self._make_layers(32, 16, 1, 1, 1)
        self.layer2 = self._make_layers(16, 24, 6, 2, 2)
        self.layer3 = self._make_layers(24, 32, 6, 2, 3)
        self.layer4 = self._make_layers(32, 64, 6, 2, 4)
        self.layer5 = self._make_layers(64, 96, 6, 1, 3)
        self.layer6 = self._make_layers(96, 160, 6, 2, 3)
        self.layer7 = self._make_layers(160, 320, 6, 1, 1)

        self.conv2 = nn.Conv2d(320, 1280, kernel_size = 1, stride = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(1280)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1280, num_classes)

    def _make_layers(self, input, output, t, s, n):
        layers = []

        layers.append(Bottleneck2(input, output, t, s))
        for i in range(n - 1):
            layers.append(Bottleneck2(output, output, t, 1))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x