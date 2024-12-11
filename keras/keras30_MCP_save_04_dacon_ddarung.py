import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)


2. 모델구성
model = Sequential()
model.add(Dense(128, activation = 'relu', input_dim=9))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 30,
    restore_best_weights=True
)
import datetime 
date = datetime.datetime.now() #데이트라는 변수에 현재 시간을 반환한다.
print(date) #2024-07-26 16:49:51.174797
print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M") #시간을 문자열로 바꿔줌
print(date) #0726_1654
print(type(date))



path1 = './_save/keras30_mcp/04_dacon_ddarung/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # '1000-0.7777.hdf5'  #fit에서 반환되는 값을 빼오는 것이다. 
filepath = "".join([path1, 'k30_', date, '_', filename])

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath 
)


hist = model.fit(x_train, y_train, epochs=1000, batch_size=2, verbose=1, 
                 validation_split=0.25, callbacks=[es, mcp])

end = time.time()

4. 평가, 예측

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

# x,y,train_size=0.9, random_state= 4343  epochs=1000, batch_size=32
# 로스 : 1687.8980712890625                 1. submission_0716_1628
# r2 스코어 :  0.69598984269193

y_submit = model.predict(test_csv)

submission_csv['count'] = y_submit

submission_csv.to_csv(path + "submission_0716_2045.csv")

print('로스 :', loss)
print("r2 스코어 : ", r2)

# print(hist)
# print('=====================hist.history======================')
# print(hist.history)
# print('=====================loss=======================')
# print(hist.history['loss'])
# print('=====================val_loss======================')
# print(hist.history['val_loss'])
# print('========================================================')

# import matplotlib.pyplot as plt
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c='red', label='loss')
# plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
# plt.legend(loc='upper right')
# plt.title('kaggle Loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid()
# plt.show()


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