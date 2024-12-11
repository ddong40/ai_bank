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

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

#1. 데이터
path = 'C:/Users/ddong40/ai_2/_data/dacon/따릉이/'       # 경로지정 #상대경로 

train_csv = pd.read_csv(path + "train.csv", index_col=0)   # . = 루트 = AI5 폴더,  index_col=0 첫번째 열은 데이터가 아니라고 해줘
print(train_csv)     # [1459 rows x 10 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0) #predict할 x데이터
print(test_csv)     # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + "submission.csv", index_col=0) #predict 할 y데이터
print(submission_csv)       #[715 rows x 1 columns],    NaN = 빠진 데이터
# 항상 오타, 경로 , shape 조심 확인 주의

print(train_csv.shape)  #(1459, 10)
print(test_csv.shape)   #(715, 9)
print(submission_csv.shape)     #(715, 1)

print(train_csv.columns)
# # ndex(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')
# x 는 결측치가 있어도 괜찮지만 y 는 있으면 안된다

train_csv.info() #train_csv의 정보를 알려주는 함수

################### 결측치 처리 1. 삭제 ###################
# print(train_csv.isnull().sum())
print(train_csv.isna().sum()) 

train_csv = train_csv.dropna() #dropna는 결측치의 행을 삭제해라

test_csv = test_csv.fillna(test_csv.mean())  #fillna 채워라 #mean함수는 뭔디    #컬럼끼리만 평균을 낸다

x = train_csv.drop(['count'], axis=1)           # drop = 컬럼 하나를 삭제할 수 있다. #axis는 축이다 
print(x)        #[1328 rows x 9 columns]
y = train_csv['count']         # 'count' 컬럼만 넣어주세요
print(y.shape)   # (1328,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                    random_state= 100)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

print(x_train.shape)
print(x_test.shape)

# (929, 9)
# (399, 9)
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).to(DEVICE)
y_test = torch.FloatTensor(y_test).to(DEVICE)
test_csv = torch.FloatTensor(test_csv).to(DEVICE)

from torch.utils.data import TensorDataset # x, y 합친다
from torch.utils.data import DataLoader 

train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_set, batch_size = 40,shuffle=True)
test_loader = DataLoader(test_set, batch_size=40, shuffle=False)

# print(y_test.size)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 32)
        self.linear4 = nn.Linear(32, 16)
        self.linear5 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, input_size): #정의된 변수 순서대로 정렬
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

model = Model(9, 1).to(DEVICE)

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
    

epochs = 1000
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


y_submit = model(test_csv).detach().cpu().numpy()


submission_csv['count'] = y_submit

submission_csv.to_csv(path + "submission_0716_2045.csv")

print('로스 :', loss)
print("r2 스코어 : ", acc)




# 스케일 하기 전
# 로스 : 2866.528076171875
# r2 스코어 :  0.610590209763064

# minmaxscaler 
# 로스 : 1709.8336181640625
# r2 스코어 :  0.767723940664879

# standardscaler
# 로스 : 2038.3023681640625
# r2 스코어 :  0.7231023809840584

# maxabscaler
# 로스 : 1899.0213623046875
# r2 스코어 :  0.7420232740873068

# RobustScaler
# 로스 : 1984.86962890625
# r2 스코어 :  0.7303610272161217

# 세이브 점수
# 로스 : 1929.4364013671875
# r2 스코어 :  0.7378915337148807

# 로스 : 2769.141845703125
# r2 스코어 :  0.6238198311394536
