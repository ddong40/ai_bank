import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용DEVICE : ', DEVICE)

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3]])
x = np.transpose(x)
#x = np.array([[1,6], [2,7], [3,8], [4,9],[ 5,10]])
y = np.array([1,2,3,4,5,6,7,7,9,10])

x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE) 
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE) 

print(x.shape, y.shape)
# (10, 2) (10,)

#2. 모델구성
model = nn.Sequential(
    nn.Linear(2,10),
    nn.Linear(10,3),
    nn.Linear(3,4),
    nn.Linear(4,3),
    nn.Linear(3,5),
    nn.Linear(5,1)
).to(DEVICE)

#3. 컴파일 훈련

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad() #
    
    hypothesis = model(x) #y=wx+b에 x를 넣에 예측한 'y값
    
    loss = criterion(hypothesis, y) #'y값과 y값의 로스, 여기까지 순전파
    
    loss.backward() #역전파 시작
    optimizer.step()
    
    return loss.item()

epochs = 1000
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch: {}, loss: {}'.format(epoch, loss))

print("=========================================================")

#4. 평가, 예측

def evaluate(model, criterion, x, y):
    model.eval()
    
    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y, y_predict)
    return loss2.item()

loss2 = evaluate(model, criterion, x, y)
print('최종 loss :', loss2)

x_pre = (torch.Tensor([[11, 1.2]]).to(DEVICE))

result = model(x_pre)

print('4의 예측 값 :', result)
print('4의 예측 값 :', result.item())