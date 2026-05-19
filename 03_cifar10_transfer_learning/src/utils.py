import random
import numpy as np
import torch
from torchvision import models
import model as md
import torch.optim as optim
from torchvision import transforms

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_model(model_name):
    if model_name == 'ResNet18':
        return models.resnet18(weights = 'IMAGENET1K_V1')
    elif model_name == 'ResNet50':
        return models.resnet50(weights = 'IMAGENET1K_V1')
    elif model_name == 'manual_ResNet18':
        return md.ResNet18_cifar()
    elif model_name == 'manual_ResNet50':
        return md.ResNet50_cifar()
    else:
        raise ValueError("Not found model")
    
    
def get_optimizer(mode, model, optimizer, lr = 1e-3):
    if mode == 'freeze':
        return optim.Adam(model.fc.parameters(), lr = 1e-3)
    elif mode == 'finetune':
        if optimizer == 'SGD':
            return optim.SGD(model.parameters(), lr = 1e-2, momentum = 0.9)
        elif optimizer == 'Adam':
            return optim.Adam(model.parameters(), lr = 1e-4)
        elif optimizer == 'AdamW':
            return optim.AdamW(model.parameters(), lr = 1e-4, weight_decay = 1e-2)
        else:
            raise ValueError("Not fount optimizer")
    else:
        return optim.Adam(model.parameters(), lr = lr)
    
def get_augmentation(aug_type):
    if aug_type == 'base':
        aug_transform = transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize( mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])
        return aug_transform
    else:
        raise ValueError("Not found augmetation")