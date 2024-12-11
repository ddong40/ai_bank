import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu' )
print('torch : ', torch.__version__, '사용DEVICE : ', DEVICE)

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target 

# x = torch.FloatTensor(x)
# y = torch.LongTensor(y) # 0,1,2만 있기 때문에 LongTensor 사용가능하다.

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True, random_state=1234, stratify=y)

print(x_train.shape, y_train.shape)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) #스케일링하면 넘파이러 변환됨

print(x_train.shape, y_train.shape)


x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)

print(x_train.shape, y_train.shape)
# torch.Size([112, 4]) torch.Size([112])

from torch.utils.data import TensorDataset # x, y 합친다
from torch.utils.data import DataLoader # batch 정의



#2. 모델 
# model = nn.Sequential(
#     nn.Linear(4, 32),
#     nn.ReLU(),
#     nn.Linear(32,32),
#     nn.ReLU(),
#     nn.Linear(32,16),
#     nn.ReLU(),
#     nn.Linear(16,3)
# ).to(DEVICE)

train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_set, batch_size = 40,shuffle=True)
test_loader = DataLoader(test_set, batch_size=40, shuffle=False)



class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64,32)
        self.linear3 = nn.Linear(32,32)
        self.linear4 = nn.Linear(32,16)
        self.linear5 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax()
    
    #순전파! 
    
    def forward(self, input_size):
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

model = Model(4, 3).to(DEVICE)
        
        
#3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss() #sparse categorical entropy과 같은거임. 이거 쓰면 원핫 인코딩 안해도 됨 softmax도 안해도 됨

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, loader):
    total_loss = 0
    
    for x_batch, y_batch in loader:
        
        # model.train 생략
        optimizer.zero_grad() #기울기 값을 0으로 둠으로 그라디언트 배니싱에 대한 문제 해결, 기울기가 누적되면서
        hypothesis = model(x_batch) # y' = wx+b
        loss = criterion(hypothesis, y_batch)
        
        loss.backward() #역전파 시작
        optimizer.step() #가중치 갱신
        total_loss += loss.item()
    return total_loss / len(loader)

EPOCHS = 1000 #특정 상수를 표기하기 위할 때에는 대문자로 표기하는 경우가 있다. #변수는 바뀌는 수, 상수는 고정된 수
for epoch in range(1, EPOCHS+1):
    loss = train(model, criterion, optimizer, train_loader) #loss 반환 순전파 역전파 하면서 갱신 이게 1 epoch
    print('epoch: {}, loss: {:.8f}'.format(epoch, loss))
    print(f'epoch: {epoch}, loss: {loss:.8f}') #같은 방식

#4. 평가 예측
def evaluate(model, criterion, loader):
    model.eval()
    
    total_loss = 0
    for x_batch, y_batch in loader:
        
        with torch.no_grad():
            y_predict = model(x_batch)
            loss = criterion(y_predict, y_batch)
            total_loss += loss.item()    
    return total_loss / len(loader)

last_loss = evaluate(model, criterion, test_loader)

print('loss : ', last_loss)

test_loader2 = DataLoader(test_set, batch_size=1, shuffle=False)

x_test = []
y_test = []

for i, a in test_loader2:
    x_test.extend(i.detach().cpu().numpy())
    y_test.extend(a.detach().cpu().numpy())

x_test = np.array(x_test)
y_test = np.array(y_test)

x_test = torch.FloatTensor(x_test).to(DEVICE)
y_test = torch.FloatTensor(y_test).to(DEVICE)


######### acc  출력해봐욤 ########
y_pred = model(x_test)

y_pred = torch.argmax(y_pred, dim=1)

# acc = accuracy_score(y_pred, y_test)

print(y_pred)

score = (y_pred == y_test).float().mean()
print('accuracy : {:.4f}'.format(score))
print(f'accuracy : {score:.4f}')
# accuracy : 1.0000

score2 = accuracy_score(y_pred.cpu().numpy(), y_test.cpu().numpy())
# accuracy : 0.9737

print(f'accuracy_score : {score2:.4f}')
