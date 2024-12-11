#29_1에서 가져옴
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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
model = Sequential()
# model.add(Dense(100, input_dim=13)) 
model.add(Dense(32, input_shape=(13,))) #백터형태로 받아들인다 백터가 어차피 column 이니까 #이미지일때는 input_shape=(8,8,1) 
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))




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
mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose = 1, #가장 좋은 지점을 알려줄 수 있게 출력함
    save_best_only=True,
    filepath = './_save/keras29_mcp/keras29_mcp3.hdf5'
)

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

# 로스 :  0.2362709939479828
# 정확도 :  0.913

# RobustScaler
# 로스 :  17.011682510375977
# r2 score : 0.8183080232863937


# 로스 :  8.781745910644531
# r2 score : 0.9062072413873566

# Epoch 00214: val_loss did not improve from 13.15108