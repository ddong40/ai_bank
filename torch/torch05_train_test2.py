import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 DEVICE :', DEVICE)

#1. 데이터
x = np.array(range(100)).transpose()
y = np.array(range(1, 101)).transpose()
x_predict = np.array([101, 102]).transpose()
x = 


####[실습] train_test_split 사용!

