import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import random 

random.seed(333)
np.random.seed(333)
torch.manual_seed(333) #토치 고정
torch.cuda.manual_seed(333) #gpu 고정

# USE_CUDA = torch.cuda.is_available()
# DEVICE = torch.device('cuda:0'if USE_CUDA else 'cpu')
# print('torch' , torch.__version__, '사용DEVICE : ', DEVICE)

DEVICE = 'cuda:0' if torch.cuda.is_available else 'cpu'
print(DEVICE)


path = '_data/kaggle/netflix-stock-prediction/'
train_csv = pd.read_csv(path + 'train.csv')
print(train_csv)
print(train_csv.info())
print(train_csv.describe())

import matplotlib.pyplot as plt 
data = train_csv.iloc[:, 1:4] # 판다스 열행인데 판다스를 행렬 형태로 쓸 수 있는 것이  loc iloc이다
# iloc 에서 i는 index loc는 location # 사용하면 열행 형태로 끌어오게 되는 판다스를 행렬로 끌어와 작업하게 된다.

data['종가'] = train_csv['Close']
print(data)
# hist = data.hist()
# plt.show()

##### 컬럼별로 최소 최대가 아닌 전체 데이터셋의 최소 최대가 적용되는 문제 ##### 
# data = train_csv.iloc[:, 1:4]
# data = (data - np.min(data)) / (np.max(data) - np.min(data))
# data = pd.DataFrame(data)
# print(data.describe())

##### 컬럼별로 최소 최대가 아닌 전체 데이터셋의 최소 최대가 적용되는 문제해결, axis=0을 넣어주면 된다 ##### 
# data = train_csv.iloc[:, 1:4]
# data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
# data = pd.DataFrame(data)
# print(data.describe())


from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data import DataLoader



class Custom_Dataset(Dataset):
    def __init__(self):
        self.csv = train_csv # self에 train_csv를 가져오겠다.
        
        self.x = self.csv.iloc[:, 1:4].values #시가, 고가, 저가 컬럼이 된다.
        # 정규화
        self.x = (self.x - np.min(self.x, axis=0)) / (np.max(self.x, axis=0)  - np.min(self.x, axis=0)) #minmax scaler
        # y데이터
        self.y = self.csv['Close'].values
    
    def __len__(self):
        return len(self.x) - 30
    
    def __getitem__(self, i):
        x = self.x[i:i+30]
        y = self.y[i+30]  
        
        return x,y
    
aaa  = Custom_Dataset()
# print(aaa)

# print(type(aaa))

# print(aaa[0])


# print(aaa[0][0].shape) #(30, 3)
# print(aaa[0][1]) #94
# print(len(aaa)) #937
# print(aaa[936])

# train_loader = DataLoader(aaa,  batch_size=32) 

# aaa = iter(train_loader)
# bbb= next(aaa)
# print(bbb)

#2. 모델

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=3, #피쳐의 개수
                          hidden_size = 64, #output node의 개수, tensorflow에선 unit
                          num_layers=5, # rnn의 계층을 쌓는 파라미터
                          batch_first = True, # 적용 시 torch에서 rnn연산시 바뀌는 순서를 정상적인 순서로 배치함
                          )
        self.fc1 = nn.Linear(in_features=30*64, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=1)
        self.relu = nn.ReLU()
        
    def forward(self, x, h0):
        x, h0 = self.rnn(x)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

model = RNN().to(DEVICE)

#3. 컴파일 훈련
from torch.optim import Adam

# optim = Adam(params=model.parameters(), lr=0.001)

import tqdm




# for epoch in range(1, 201):
#     iterator = tqdm.tqdm(train_loader)
#     for x, y in iterator:
#         optim.zero_grad()
        
#         h0 = torch.zeros(5, x.shape[0], 64 ).to(DEVICE) #(num_layers, batch_size, hidden size) = (5, 32, 64)
        
#         hypothesis = model(x.type(torch.FloatTensor).to(DEVICE), h0)
#         # type함수로 형태 변형 가능
#         loss = nn.MSELoss()(hypothesis, y.type(torch.FloatTensor).to(DEVICE))
        
#         loss.backward()
#         optim.step()

#         iterator.set_description(f'epoch:{epoch} loss:{loss.item()}')

save_path = './/_save//torch//'
# torch.save(model.state_dict(), save_path + 't22.pth')


from sklearn.metrics import r2_score

#4. 평가 예측

train_loader = DataLoader(aaa,  batch_size=1) 


x_test2 = []
y_test2 = []
y_predict = []
total_loss = 0

with torch.no_grad():
    model.load_state_dict(torch.load(save_path+'t22.pth', map_location=DEVICE,
                                     weights_only=True,
                                     ))
    
    for x_test, y_test in train_loader:
        h0 = torch.zeros(5, x_test.shape[0], 64).to(DEVICE)
        
        y_pred = model(x_test.type(torch.FloatTensor).to(DEVICE), h0)
        y_predict.append((y_pred.item()))
        
        loss = nn.MSELoss()(y_pred,
                            y_test.type(torch.FloatTensor).to(DEVICE))
        
        total_loss += loss/ len(train_loader)
  
        
# print(f'y_predict : {y_predict}, \n shape : {y_predict.view}')
print('total_loss : ', total_loss.item())


for i, a in train_loader:
    x_test2.extend(a.detach().cpu().numpy())
    y_test2.extend(a.detach().cpu().numpy())        


r2 = r2_score(y_predict, y_test2)  

print(r2)