import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용DEVICE : ', DEVICE)

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3],
              [10,9,8,7,6,5,4,3,2,1]])
x = np.transpose(x)
#x = np.array([[1,6], [2,7], [3,8], [4,9],[ 5,10]])
y = np.array([1,2,3,4,5,6,7,7,9,10])

x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE) 
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE) 

print(x.shape, y.shape)

### 맹그러봐!!!
#예측값 : [10, 1.3, 1]

#2. 모델

model = nn.Sequential(
    nn.Linear(3,5),
    nn.Linear(5,4),
    nn.Linear(4,5),
    nn.Linear(5,3),
    nn.Linear(3,1)
).to(DEVICE)

#3.컴파일, 훈련

criterion = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad() #각 배치마다 기울기를 0으로 초기화하여 , 기울기 누적에 의한 문제를 해결한다.
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

epochs = 1000
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch: {}, loss:{}'.format(epoch, loss))
    
print('===========================================================')

# 4. 평가 예측

def evaluate(model, criterion, x,  y):
    model.eval()
    
    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y, y_predict)
    return loss2.item()

loss2 = evaluate(model, criterion, x, y)
print('최종 loss :', loss2)

x_pre = (torch.Tensor(([10, 1.3, 1])).to(DEVICE))
print(x_pre)

result = model(x_pre)

print("4의 예측값 :", result)           
print("4의 예측값 :", result.item()) 

# 4의 예측값 : tensor([5.3377], device='cuda:0', grad_fn=<ViewBackward0>)
# 4의 예측값 : 5.337657451629639