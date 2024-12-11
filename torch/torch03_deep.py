import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용DEVICE : ', DEVICE)
# torch :  2.4.1+cu124 사용DEVICE :  cuda


#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

x = torch.FloatTensor(x)
print(x.shape) #torch.Size([3])
print(x.size()) #torch.Size([3])

x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE) # (3,) -> (3,1)

print(x)
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE) # (3,) -> (3,1)

print(x.shape, y.shape)
print(x.size(), y.size())

#2. 모델 구성
# model = Sequential()
# model.add(Dense(1, input_dim=1))
# model = nn.Linear(1, 1).to(DEVICE) #인풋, 아웃풋 # y = xw + b
model = nn.Sequential(
    nn.Linear(1,5),
    nn.Linear(5,4),
    nn.Linear(4,3),
    nn.Linear(1,5),
).to(DEVICE)



#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
criterion = nn.MSELoss() #로스의 표준은 MSE
# optimizer = optim.Adam(model.parameters(), lr=0.01)
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y): 
    # model.train()       # 훈련모드
    optimizer.zero_grad() # 각 배치마다 기울기를 초기화하여, 0으로 초기화함, 기울기 누적에 의한 문제 해결. 경사하강법의 경사를 말하는 것이다.
    
    hypothesis = model(x) # y = wx + b
    
    loss = criterion(hypothesis, y) #loss = mse() hypothesis가 예측값임
    
    loss.backward() # 기울기(gradient)값 계산까지. #역전파 시작하려고 준비하는 과정
    
    optimizer.step() # 가중치(w) 갱신 즉 로스 갱신 # 역전파 끝! 왜? 
    
    return loss.item() #텐서로 되어있기 때문에 우리가 사용할 수 있는 값으로 변환

epochs = 2000

for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch : {}, loss: {}'.format(epochs, loss)) #verbose

print('===============================================================')

#4. 평가, 예측

# loss = model.evaluate(x,y)
def evaluate(model, criterion, x, y):
    model.eval() #평가모드 #가중치 갱신을 이후에 하지 않도록 하는 것
    
    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y, y_predict)
    return loss2.item() 

loss2 = evaluate(model, criterion, x, y)
print('최종 loss : ', loss2)

results = model(torch.Tensor([[4]])).to(DEVICE)
print('4의 예측값 : ', results.item())


#데이터와 모델에만 gpu처리 해주면된다.