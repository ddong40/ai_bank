#18_1에서 가져옴
import numpy as np
import pandas as pd
import sklearn as sk
print(sk.__version__)   # 0.24.2
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time

#1. 데이터
dataset = load_boston()
print(dataset)
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x.shape)    
print(y.shape)    

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7,
                                                    random_state=6666)

# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = RobustScaler()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train) #2줄을 한 줄로 줄일 수 있다. 
x_test = scaler.transform(x_test)

print(x_train)
print(np.min(x_train), np.max(x_train)) #0.0 1.0000000000000002
print(np.min(x_test), np.max(x_test)) #-0.008298755186722073 1.1478180091225068


#2. 모델구성
model = Sequential()
# model.add(Dense(100, input_dim=13)) 
model.add(Dense(128, input_shape=(13,))) #백터형태로 받아들인다 백터가 어차피 column 이니까 #이미지일때는 input_shape=(8,8,1) 
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=1, validation_split = 0.3)
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print('로스 : ', loss)
print('r2 score :', r2)

# 로스 :  0.2362709939479828
# 정확도 :  0.913

# RobustScaler
# 로스 :  17.011682510375977
# r2 score : 0.8183080232863937