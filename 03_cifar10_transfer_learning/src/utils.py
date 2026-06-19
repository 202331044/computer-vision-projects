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

def get_augmentation():
    aug_transform = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize( mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    ])
    
    return aug_transform


MODEL_DICT = {'resnet18': lambda: models.resnet18(weights = 'IMAGENET1K_V1'),
               'resnet50': lambda: models.resnet50(weights = 'IMAGENET1K_V1'),
               'resnet18_manual': lambda: md.ResNet18CIFAR(),
               'resnet50_manual': lambda: md.ResNet50CIFAR(),
               'mobilenetv2_manual' : lambda : md.MobileNetV2(num_classes = 10, cifar = True),
               'mobilenetv2': lambda : models.mobilenet_v2(weights = models.MobileNet_V2_Weights.DEFAULT)}
    
def get_model(model_name):
    if model_name not in MODEL_DICT:
        raise ValueError(f"Not found model: {model_name}")
    
    return MODEL_DICT[model_name]()


OPTIM_DICT = {'adam': lambda params, lr: optim.Adam(params, lr = lr),
              'sgd': lambda params, lr: optim.SGD(params, lr = lr, momentum = 0.9)}

def get_optimizer(params, opt, lr = 1e-3):
    if opt not in OPTIM_DICT:
        raise ValueError(f"Not found optimizer: {opt}")
    
    return OPTIM_DICT[opt](params, lr)