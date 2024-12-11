import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu' )
print('torch : ', torch.__version__, '사용DEVICE : ', DEVICE)

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)

# print(np.array(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1234, shuffle=True, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


print(x_train.shape, y_train.shape)
# (133, 13) (133,)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)

print(y_train.shape)

#2. 모델 
model = nn.Sequential(
    nn.Linear(13, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
).to(DEVICE)

#3. 컴파일 훈련
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x_train, y_train):
    #model.train 생략
    optimizer.zero_grad() #기울기 값 0으로 둠
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)
    
    loss.backward() # 역전파 시작
    optimizer.step() #r가중치 갱신
    return loss.item()

EPOCHS = 1000
for epoch in range(1, EPOCHS+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch : {}, loss: {:.8f}'.format(epoch, loss))
    print(f'epoch: {epoch}, loss : {loss:.8f}') #위와 같은 값 출력

#4. 평가 예측
def evaluate(model, criterion, x_test, y_test):
    model.eval()
    
    with torch.no_grad(): #기울기 갱신 될 수도 있어서 미연에 방지
        hypothesis = model(x_test)
        loss = criterion(hypothesis, y_test)
        
        return loss.item()
loss = evaluate(model, criterion, x_test, y_test)
print('loss : ', loss)

y_pred = model(x_test)

y_pred = torch.argmax(y_pred, dim=1)

print(y_pred)

score = (y_pred == y_test).float().mean()
print('accuracy : {:.4f}'.format(score))
print(f'accuracy : {score:.4f}')

score2 = accuracy_score(y_pred.cpu().numpy(), y_test.cpu().numpy())

print(f'accuracy_score : {score2:.4f}')

# accuracy_score : 0.9778