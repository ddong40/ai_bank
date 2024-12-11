import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score

USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu' )
print('torch : ', torch.__version__, '사용DEVICE : ', DEVICE)

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target 

# print(np.array(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True, random_state=1234)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).to(DEVICE)
y_test = torch.FloatTensor(y_test).to(DEVICE)

from torch.utils.data import TensorDataset # x, y 합친다
from torch.utils.data import DataLoader 

train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_set, batch_size = 40,shuffle=True)
test_loader = DataLoader(test_set, batch_size=40, shuffle=False)

print(x_train.shape, y_train.shape)

#2. 모델 구성

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 32)
        self.linear4 = nn.Linear(32, 16)
        self.linear5 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    #순전파
    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.linear5(x)
        return x

model = Model(8, 1).to(DEVICE)

#3. 컴파일 훈련

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr = 0.01)

def train(model, criterion, optimizer, loader):
    # model.train() 
    total_loss = 0
    
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        return total_loss / len(loader)

epochs = 200
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, train_loader)
    print('epoch: {}, loss : {}'.format(epoch, loss))

#4. 평가 예측

def evaluate(model, criterion, loader):
    model.eval()
    
    total_loss = 0
    for x_batch, y_batch in loader:
        with torch.no_grad():
            y_predict = model(x_batch)
            loss2 = criterion(y_batch, y_predict)
            total_loss += loss2.item()
        return total_loss / len(loader)

last_loss = evaluate(model, criterion, test_loader)
print('최종 loss : ', last_loss)        

from sklearn.metrics import accuracy_score, r2_score
test_loader2 = DataLoader(test_set, batch_size=171, shuffle=False)

x_test = []
y_test = []

for i, a in test_loader2:
    x_test.extend(i.detach().cpu().numpy())
    y_test.extend(a.detach().cpu().numpy())

x_test = np.array(x_test)
y_test = np.array(y_test)
print(x_test.shape)

x_test = torch.FloatTensor(x_test).to(DEVICE)
y_test = torch.FloatTensor(y_test).to(DEVICE)

y_pred = model(x_test)

print(y_pred.shape)
        
acc = r2_score(y_test.detach().cpu().numpy(), y_pred.detach().cpu().numpy())   

print(acc)