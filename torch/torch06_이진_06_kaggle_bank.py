import torch as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu' )
print('torch : ', torch.__version__, '사용DEVICE : ', DEVICE)


path = 'C:/Users/ddong40/ai_2/_data/kaggle/playground-series-s4e1/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
sampleSubmission = pd.read_csv(path + 'sample_submission.csv', index_col = 0)

print(train_csv.shape) 
print(test_csv.shape)

# encoder = LabelEncoder()

encoder = LabelEncoder()

print(test_csv.value_counts)
print(train_csv.value_counts)


test_csv = test_csv.drop(['CustomerId','Surname'], axis=1)



test_csv['Geography'] = encoder.fit_transform(test_csv['Geography'])
test_csv['Gender'] = encoder.fit_transform(test_csv['Gender'])
train_csv['Geography'] = encoder.fit_transform(train_csv['Geography'])
train_csv['Gender'] = encoder.fit_transform(train_csv['Gender'])

x = train_csv.drop(['CustomerId','Surname','Exited'], axis=1)
y = train_csv['Exited']

# from sklearn.preprocessing import MinMaxScaler
# scalar=MinMaxScaler()
# x[:] = scalar.fit_transform(x[:])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, shuffle= True, random_state= 512, stratify=y)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)

print(x_train.shape)
print(x_test.shape)

print(x_train)

x_train = x_train.values
x_test = x_test.values
y_train = y_train.values
y_test = y_test.values
test_csv = test_csv.values

print(x_train.shape)
print(x_test.shape)

print(y_train.shape)



x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)
test_csv = torch.FloatTensor(test_csv).to(DEVICE)


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(test_csv.shape)



#2. 모델구성
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.Linear(16, 1),
    nn.Sigmoid()
).to(DEVICE)

#3. 컴파일 훈련

criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr = 0.01)

def train(model, criterion, optimizer, x, y):
    # model.train() # 훈련모드, 디폴트
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    
    loss.backward() ##기울기 gradient 값 계산까지, 역전파 시작! 이거 안하면 처음의 weight로 계속 
    optimizer.step() #가중치(w) 갱신 #역전파 끝
    return loss.item()

epochs = 200
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch: {}, loss : {}'.format(epoch, loss)) #verbose

print('==================================================================')

#4. 평가, 예측
# loss = model.evaluate(x,y)

def evaluate(model, criterion, x, y):
    model.eval() #평가모드 // 역전파 x, 가중치 갱신 x, 기울기 계산 할 수도 안 할 수도 
                 # 드롭아웃, 배치노말 x
    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y, y_predict)
    return loss2.item()

last_loss = evaluate(model, criterion, x_test, y_test)
print('최종 loss : ', last_loss)

y_pred = model(x_test)

print(y_pred.shape)



############################################ 요 밑에 완성할 것 #################################################

from sklearn.metrics import accuracy_score, r2_score

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
