import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용DEVICE : ', DEVICE)

#1. 데이터

x = np.array([range(10)])
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3],
              [10,9,8,7,6,5,4,3,2,1]])
x = np.transpose(x)
y = np.transpose(y)
#x = np.array([[1,6], [2,7], [3,8], [4,9],[ 5,10]])


x = torch.FloatTensor(x).to(DEVICE) 
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE) 

print(x.shape, y.shape)

##맹그러봐
#예측값 : [10]

#2. 모델 구성

model = nn.Sequential(
    nn.Linear(1, 128),
    nn.Linear(128, 64),
    nn.Linear(64, 64),
    nn.Linear(64, 32),
    nn.Linear(32, 32),
    nn.Linear(32, 3)
).to(DEVICE)

#3. 컴파일 훈련
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(),lr=0.01)

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad() #각 배치마다 기울기를 0으로 초기화한다.
    
    hypothesis = model(x)
    
    loss = criterion(hypothesis, y)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

epochs = 1000
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch: {}, loss : {}'.format(epoch, loss))
    
print('=================================================')

#4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()
    
    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y, y_predict)
    return loss2.item()

loss2 = evaluate(model, criterion, x, y)
print('최종 loss : ', loss2)

x_pre = (torch.Tensor([[10]]).to(DEVICE))
print(x_pre)

result = model(x_pre)


print("4의 예측값 :", result.detach())  # 4의 예측값 : tensor([[5.5000, 1.3300, 5.5000]], device='cuda:0')


# print("4의 예측값 :", result.detach().numpy())  # numpy는 cpu에서만 돌아서 에러야

print("4의 예측값 :", result.detach().cpu().numpy())  # 4의 예측값 : [[5.499999  1.3299998 5.499999 ]]



