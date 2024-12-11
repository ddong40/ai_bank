import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import CIFAR100

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0'if USE_CUDA else 'cpu')
print('torch' , torch.__version__, '사용DEVICE : ', DEVICE)

path = './study/torch/_data'


