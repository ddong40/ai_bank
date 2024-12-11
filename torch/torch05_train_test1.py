import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 DEVICE :', DEVICE)
#1. 데이터

x_train = np.array([1,2,3,4,5,6,7]).transpose()
y_train = np.array([1,2,3,4,5,6,7]).transpose()
x_test = np.array([8,9,10,11]).transpose()
y_test = np.array([8,9,10,11]).transpose()
x_predict = np.array([12,13,14]).transpose()
x_train = torch.FloatTensor(x_train).unsqueeze(1).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

x_predict = torch.FloatTensor(x_predict).unsqueeze(1).to(DEVICE)

print(x_train.shape, y_train.shape)
# torch.Size([7, 1]) torch.Size([7, 1])
print(x_predict.shape)
# torch.Size([3, 1])



######################################

#2. 모델

model = nn.Sequential(
    nn.Linear(1,5),
    nn.Linear(5,4),
    nn.Linear(4,3),
    nn.Linear(3,1)
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer , x, y):
    optimizer.zero_grad()
    
    hypothesis = model(x)
    
    loss = criterion(hypothesis, y)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

epochs = 1000
for epoch in range(1, epochs+1):
    loss= train(model, criterion, optimizer, x_train, y_train)
    print('epoch : {}, loss : {}'.format(epoch, loss))
print('======================================================')

#4. 평가, 예측

def evaluate(model, criterion, x, y):
    model.eval()
    
    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y, y_predict)
    return loss2.item()

loss2= evaluate(model, criterion, x_test, y_test)
print('최종 loss : ', loss2)


result = model(x_predict)

print(result.detach().cpu().numpy())