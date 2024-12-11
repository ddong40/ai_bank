import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.datasets import load_wine
import time

#1 데이터
datasets = load_wine()

x = datasets.data
y = datasets.target

print(datasets)
print(datasets.DESCR)

from sklearn.preprocessing import OneHotEncoder
y = y.reshape(-1,1)
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, shuffle= True, 
                                                    random_state= 150, stratify=y)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape) #124,13
print(x_test.shape) #54, 13

x_train = x_train.reshape(124, 13, 1, 1)
x_test = x_test.reshape(54, 13, 1, 1)

# #2 모델구성

# # model = Sequential()
# # model.add(Dense(128, 'relu', input_dim=13))
# # model.add(Dropout(0.3))
# # model.add(Dense(128, 'relu'))
# # model.add(Dropout(0.3))
# # model.add(Dense(128, 'relu'))
# # model.add(Dropout(0.3))
# # model.add(Dense(128, 'relu'))
# # model.add(Dropout(0.3))
# # model.add(Dense(128, 'relu'))
# # model.add(Dropout(0.3))
# # model.add(Dense(64, 'relu'))
# # model.add(Dropout(0.3))
# # model.add(Dense(64, 'relu'))
# # model.add(Dropout(0.3))
# # model.add(Dense(64, 'relu'))
# # model.add(Dropout(0.3))
# # model.add(Dense(64, 'relu'))
# # model.add(Dropout(0.3))
# # model.add(Dense(64, 'relu'))
# # model.add(Dropout(0.3))
# # model.add(Dense(3, 'softmax'))

# input1 = Input(shape=(13))
# dense1 = Dense(128, activation = 'relu')(input1)
# drop1 = Dropout(0.3)(dense1)
# dense2 = Dense(128, activation = 'relu')(drop1)
# drop2 = Dropout(0.3)(dense2)
# dense3 = Dense(128, activation = 'relu')(drop2)
# drop3 = Dropout(0.3)(dense3)
# dense4 = Dense(128, activation = 'relu')(drop3)
# drop4 = Dropout(0.3)(dense4)
# dense5 = Dense(128, activation = 'relu')(drop4)
# drop5 = Dropout(0.3)(dense5)
# dense6 = Dense(64, activation = 'relu')(drop5)
# drop6 = Dropout(0.3)(dense6)
# dense7 = Dense(64, activation = 'relu')(drop6)
# drop7 = Dropout(0.3)(dense7)
# dense8 = Dense(64, activation = 'relu')(drop7)
# drop8 = Dropout(0.3)(dense8)
# dense9 = Dense(64, activation = 'relu')(drop8)
# drop9 = Dropout(0.3)(dense9)
# dense10 = Dense(64, activation = 'relu')(drop9)
# drop10 = Dropout(0.3)(dense10)
# output1 = Dense(3, activation = 'softmax')(drop10)
# model = Model(inputs = input1, outputs = output1)

model = Sequential()
model.add(Conv2D(32, 2, activation='relu', input_shape=(13,1,1), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32 ,2, activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32 ,2, activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32 ,2, activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(32, activation='relu', input_shape=(32,)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation = 'softmax'))



#컴파일 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
start_time = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 50,
    restore_best_weights= True)

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path1 = './_save/keras39/09_wine/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([path1, 'k30_', date, '_', filename])

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose =1,
    save_best_only=True,
    filepath = filepath
)


model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.25, callbacks=[es, mcp])
end_time = time.time()

#평가 예측
loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)

y_test = np.argmax(y_test).reshape(-1, 1)
y_pred = np.argmax(y_pred).reshape(-1, 1)

acc = accuracy_score(y_test, y_pred)
print("로스값 : ", loss[0])
print("accuracy : ", acc)
print("걸린시간 : ", round(end_time - start_time, 2), "초" )

# print(y_pred)

# 로스값 :  0.2663238048553467
# accuracy :  0.907

# minmaxscaler
# 로스값 :  0.2635650038719177
# accuracy :  0.963

# standardscaler
# 로스값 :  0.21828098595142365
# accuracy :  0.981

# MaxAbsScaler
# 로스값 :  0.9603663682937622
# accuracy :  0.944

# RobustScaler
# 로스값 :  0.23232993483543396
# accuracy :  0.963

# 세이브 값
# 로스값 :  0.3106008470058441
# accuracy :  0.963

# drop out
# 로스값 :  0.4955110251903534
# accuracy :  0.944

# cpu
# 로스값 :  1.1234495639801025
# accuracy :  0.944
# 걸린시간 :  5.12 초


# gpu
# 로스값 :  1.2003084421157837
# accuracy :  0.944
# 걸린시간 :  38.69 초

# dnn - cnn으로 변환
# 로스값 :  0.494925856590271
# accuracy :  1.0
# 걸린시간 :  37.4 초