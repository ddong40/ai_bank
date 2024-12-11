import torch as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu' )
print('torch : ', torch.__version__, '사용DEVICE : ', DEVICE)

datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=336, stratify=y)

print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape) 
print(x_train)

x_train = torch.FloatTensor(x_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test).unsqueeze(1).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

print(x_train.shape, y_train.shape) #torch.Size([309, 1, 10]) torch.Size([309, 1])
print(x_test.shape, y_test.shape) #torch.Size([133, 1, 10]) torch.Size([133, 1])

#2. 모델 

# model = nn.Sequential(
#     nn.Linear()
# )




# criterion 