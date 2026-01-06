"""
mnist 손글씨 숫자 분류를 위한 Pytorch 구현
""" 

import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms 
import matplotlib.pyplot as plt 
import numpy as np 

# 하이퍼 파라미터 생성

BATCH_SIZE = 64 
LEARNING_RATE = 0.001 
NUM_EPOCHS = 50 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"사용중인 디바이스: {DEVICE}")

class MNIST_ResNet(nn.Module):
    """
    MNIST 분류를 위한 CNN 모델
    """

    def __init__(self):
        super(MNIST_ResNet, self).__init__()
        # 첫 번째 Convolution block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        # 두 번째 Convolution block 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(128,128, kernel_size=3, padding='same')
        self.pool2 = nn.MaxPool2d(2, 2)

        # Drop out layer
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # fully connected layer
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)

        # Residual block
        self.res1 = nn.Sequential()
        self.res1.add_module('conv3', nn.Conv2d(64, 128, kernel_size=3, padding='same'))

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)


