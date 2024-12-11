# import tensorflow as tf

# gpus = tf.config.experimental.list_physical_devices('GPU')

# if(gpus) : 
#     print("쥐피유 돈다!!!")
# else:
#     print("쥐피유 없다!!!")


#29_5에서 가져옴
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
import time

#1. 데이터
dataset = load_boston()

x = dataset.data
y = dataset.target
 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7,
                                                    random_state=6666)

# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train) #2줄을 한 줄로 줄일 수 있다. 
x_test = scaler.transform(x_test)



#2. 모델구성
# model = Sequential()
# # model.add(Dense(100, input_dim=13)) 
# model.add(Dense(64, input_shape=(13,))) #백터형태로 받아들인다 백터가 어차피 column 이니까 #이미지일때는 input_shape=(8,8,1) 
# model.add(Dropout(0.3)) #30 percenet를 빼서 훈련을 시키지 않는다. 상위 레이어에서 drop out한다는 뜻
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.3)) # 드롭아웃은 통상 0.5까지
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2)) 
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.1)) 
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1))

input1 = Input(shape=(13))
dense1 = Dense(64, name='ys1')(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(64, name='ys2', activation = 'relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(32, name='ys3', activation = 'relu')(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(32, name= 'ys4', activation = 'relu')(drop3)
drop4 = Dropout(0.1)(dense4)
dense5 = Dense(16, name= 'ys5', activation = 'relu')(drop4)
output1 = Dense(1)(dense5)
model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 10,
    verbose = 1,
    restore_best_weights=True
)

#################mcp 세이브 파일명 만들기 시작##################

import datetime 
date = datetime.datetime.now() #데이트라는 변수에 현재 시간을 반환한다.
print(date) #2024-07-26 16:49:51.174797
print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M") #시간을 문자열로 바꿔줌
print(date) #0726_1654
print(type(date))


path = './_save/keras32/01_boston'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # '1000-0.7777.hdf5'  #fit에서 반환되는 값을 빼오는 것이다. 
filepath = "".join([path, 'k32_', date, '_', filename])

############mcp세이브 파일명 만들기 끝################

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose = 1, #가장 좋은 지점을 알려줄 수 있게 출력함
    save_best_only=True,
    filepath = filepath
)
# 생성 예" './_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5'


start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=1, validation_split = 0.3, callbacks=[es, mcp])
end = time.time()

# model.save('./_save/keras29_mcp/keras29_3_save_model.h5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print('로스 : ', loss)
print('r2 score :', r2)
print('시간 : ', round(end-start, 3), '초')

# 로스 :  0.2362709939479828
# 정확도 :  0.913

# RobustScaler
# 로스 :  17.011682510375977
# r2 score : 0.8183080232863937


# 로스 :  8.781745910644531
# r2 score : 0.9062072413873566

# Epoch 00214: val_loss did not improve from 13.15108

# drop out..
# 로스 :  15.432994842529297
# r2 score : 0.8351691183396657

# 함수
# 로스 :  16.757686614990234
# r2 score : 0.821020823871225

# cpu
# 로스 :  17.194480895996094
# r2 score : 0.8163556945265341
# 시간 :  1.748 초

# gpu
# 로스 :  15.881122589111328
# r2 score : 0.8303829218305986
# 시간 :  3.548 초