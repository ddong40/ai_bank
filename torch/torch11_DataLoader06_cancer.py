import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu' )
print('torch : ', torch.__version__, '사용DEVICE : ', DEVICE)

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=369, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE) #실수형으로 변환
# x_train = torch.DoubleTensor(x_train).to(DEVICE)

x_test = torch.FloatTensor(x_test).to(DEVICE)
# x_test = torch.DoubleTensor(x_test).to(DEVICE)


y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
# y_train = torch.LongTensor(y_train).unsqueeze(1),to(DEVICE) 
# y_train = torch.IntTensor(y_train).unsqueeze(1),to(DEVICE) #정수형으로 변환

y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)
# y_test = torch.LongTensor(y_test).unsqueeze(1).to(DEVICE)
# y_test = torch.IntTensor(y_test).unsqueeze(1).to(DEVICE)

print('======================================================')
print(x_train.shape, x_test.shape) #torch.Size([398, 30]) torch.Size([171, 30])
print(y_train.shape, y_test.shape) #torch.Size([398, 1]) torch.Size([171, 1])
print(type(x_train), type(y_train)) #<class 'torch.Tensor'> <class 'torch.Tensor'>

from torch.utils.data import TensorDataset # x, y 합친다
from torch.utils.data import DataLoader # batch 정의

# 토치 데이터셋 만들기 1. x와 y를 합친다.
train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)
print(train_set) #<torch.utils.data.dataset.TensorDataset object at 0x0000026A04881250>
print(type(train_set)) #<class 'torch.utils.data.dataset.TensorDataset'>
print(len(train_set)) #398는 행의 개수
print(train_set[0])
# (tensor([ 0.1424, -1.2523,  0.2387, -0.0102,  0.4852,  1.4386,  0.6534,  0.3351,
#          0.9607,  1.6571,  0.5107,  0.5140,  0.9537,  0.2175,  1.0286,  1.4196,
#          0.6065,  0.6138,  0.7285,  0.6069,  0.0289, -1.1929,  0.1951, -0.1322,
#         -0.0354,  0.7077,  0.2334, -0.0659, -0.0966,  0.5060], device='cuda:0'), tensor([1.], device='cuda:0'))
print(train_set[0][0]) # 첫 번째 x
print(train_set[0][1]) # 첫 번째 y train_set[397] 까지 있음

#토치 데이터셋 만들기 2. batch 넣어준다. 끝! 
train_loader = DataLoader(train_set, batch_size = 40,shuffle=True)
test_loader = DataLoader(test_set, batch_size=40, shuffle=False)
print(len(train_loader)) #10 10개인 이유는 batch를 40으로 나눠줬기 때문에 
print(train_loader) #<torch.utils.data.dataloader.DataLoader object at 0x000002123AD35160>
# print(train_loader[0]) #TypeError: 'DataLoader' object is not subscriptable #이터레이터라서 리스트 볼 때 처럼 보려고하면 에러가 생김
print('===================================================================================')
# 1. 이터레이터 for문 확인

# for aaa in train_loader:
#     print(aaa)
#     break #첫번째만 나옴

bbb = iter(train_loader)
# aaa = bbb.next() # 파이썬 3.9이후로 안 먹힘
aaa = next(bbb)
print(aaa)
# AttributeError: '_SingleProcessDataLoaderIter' object has no attribute 'next'


#2. 모델구성
# model = nn.Sequential(
#     nn.Linear(30, 64),
#     nn.ReLU(),
#     nn.Linear(64, 32),
#     nn.ReLU(),
#     nn.Linear(32, 32),
#     nn.ReLU(),
#     nn.Linear(32, 16),
#     nn.Linear(16, 1),
#     nn.Sigmoid()
# ).to(DEVICE)

class Model(nn.Module): #클래스 선언
    def __init__(self, input_dim, output_dim):
        # super().__init__() #nn.Midule에 있는 함수를 상속받아서 모두 쓰겠다. super는 default이다. 
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 32)
        self.linear4 = nn.Linear(32, 16)
        self.linear5 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
    
    #순전파!      
    def forward(self, input_size): #정의된 변수를 순서대로 정렬해 줌
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.linear5(x)
        x = self.sigmoid(x)
        return x         

model = Model(30, 1).to(DEVICE) #클래스 인스턴스화 #선언해준 input_dim과 outpu_dim 삽입

#3. 컴파일 훈련

criterion = nn.BCELoss() 

optimizer = optim.Adam(model.parameters(), lr = 0.01)

def train(model, criterion, optimizer, loader):
    # model.train() # 훈련모드, 디폴트
    total_loss = 0
    
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch) #기울기 계산은 batch 단위로 진행되기에 batch 안에 들어가야함
    
        loss.backward() ##기울기 gradient 값 계산까지, 역전파 시작! 이거 안하면 처음의 weight로 계속 
        optimizer.step() #가중치(w) 갱신 #역전파 끝
        total_loss += loss.item()
        return total_loss / len(loader) #10번의 값을 10으로 나눠줘야함

epochs = 200
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, train_loader)
    print('epoch: {}, loss : {}'.format(epoch, loss)) #verbose

print('==================================================================')

#4. 평가, 예측
# loss = model.evaluate(x,y)

def evaluate(model, criterion, loader):
    model.eval() #평가모드 // 역전파 x, 가중치 갱신 x, 기울기 계산 할 수도 안 할 수도 
                 # 드롭아웃, 배치노말 x
    total_loss = 0
    for x_batch, y_batch in loader:
        
        with torch.no_grad():
            y_predict = model(x_batch)
            loss2 = criterion(y_batch, y_predict)
            total_loss += loss2.item()
        return total_loss / len(loader)

last_loss = evaluate(model, criterion, test_loader)
print('최종 loss : ', last_loss)



############################################ 요 밑에 완성할 것 (데이터 로더 사용하는 것으로 바꿔) ########################################

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

y_test = torch.round(y_test)
y_pred = torch.round(y_pred)

y_test = y_test.detach().cpu().numpy()
y_pred = y_pred.detach().cpu().numpy()


# y_test = np.reshape(y_test, (171, 1))
# y_pred = np.reshape(y_pred, (171, 1))
print(y_test.shape)
print(y_pred.shape)

# print(type(y_test))

# y_test = np.round(y_test)
# y_pred = np.round(y_pred)



acc = accuracy_score(y_test, y_pred)

# acc = acc.detach().cpu().numpy()

print('acc 스코어 : ', acc)

