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

#1. 데이터
path = 'C:/Users/ddong40/ai/_data/bike-sharing-demand/' #절대경로(경로가 풀로 다 들어간 경우)
# path = 'C:/Users/ddong40/ai/_data/bike-sharing-demand' #위와 다 같음
# path = 'C://Users//ddong40//ai//_data//bike-sharing-demand' #위와 다 같음

train_csv = pd.read_csv(path + "train.csv", index_col = 0)
test_csv = pd.read_csv(path + "test.csv", index_col = 0)
sampleSubmission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.shape) # (10886, 11)
print(test_csv.shape) # (6493, 8)
print(sampleSubmission.shape) #(6493, 1)

print(train_csv.columns)
# ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    #    'humidity', 'windspeed', 'casual', 'registered', 'count'],
    #   dtype='object'
    
print(train_csv.info())
print(test_csv.info())

print(train_csv.describe())

######### 결측치 확인 ###########

print(train_csv.isna().sum())
print(train_csv.isnull().sum())
print(test_csv.isna().sum())
print(test_csv.isnull().sum())


###### x와 y분리
x = train_csv.drop(['casual','registered','count'], axis=1) #이 리스트의 컬럼들을 axis 1에 넣어 드랍해주세요 라는 뜻
print(x)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=10)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
test_csv = np.array(test_csv)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).to(DEVICE)
y_test = torch.FloatTensor(y_test).to(DEVICE)
test_csv = torch.FloatTensor(test_csv).to(DEVICE)

print(x_train.shape)
print(y_train.shape)


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

model = Model(8, 1).to(DEVICE)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr = 0.01)

def train(model, criterion, optimizer, x, y):
    # model.train() 
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    
    loss.backward()
    optimizer.step()
    return loss.item()

epochs = 1000
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch: {}, loss : {}'.format(epoch, loss))

#4. 평가 예측

def evaluate(model, criterion, x, y):
    model.eval()
    
    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y, y_predict)
    return loss2.item()

last_loss = evaluate(model, criterion, x_test, y_test)
print('최종 loss : ', last_loss)        

y_pred = model(x_test)

print(y_pred.shape)
        
acc = r2_score(y_test.detach().cpu().numpy(), y_pred.detach().cpu().numpy())   

print(acc)


y_submit = model(test_csv).detach().cpu().numpy()


#5. 파일 출력
sampleSubmission['count'] = y_submit
print(sampleSubmission)

sampleSubmission.to_csv(path+'samplesubmission_0717_1413.csv') #to_csv는 이 데이터를 ~파일을 만들어서 거기에 넣어줄거임
print(loss)
