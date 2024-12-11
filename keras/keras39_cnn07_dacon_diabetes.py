import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten
import sklearn as sk
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import time

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=250)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(x_train.shape) 399, 10
# print(x_test.shape) 133, 10


x_train = x_train.reshape(309,5,2,1)
x_test = x_test.reshape(133,5,2,1)




# print(x)
# print(y)
# print(x.shape, y.shape)   # (442, 10) (442,)
# 분류 데이터는 0과 1만 있음 y값이 종류가 많으면 폐기모델

# [실습]
# R2 0.62 이상 -0.1 (0.52)

# #2. 모델구성
# model = Sequential()
# model.add(Dense(128, input_dim=10))
# model.add(Dropout(0.3))
# model.add(Dense(128))
# model.add(Dropout(0.3))
# model.add(Dense(128))
# model.add(Dropout(0.3))
# model.add(Dense(64))
# model.add(Dropout(0.3))
# model.add(Dense(64))
# model.add(Dropout(0.3))
# model.add(Dense(64))
# model.add(Dropout(0.3))
# model.add(Dense(64))
# model.add(Dropout(0.3))
# model.add(Dense(32))
# model.add(Dropout(0.3))
# model.add(Dense(32))
# model.add(Dropout(0.3))
# model.add(Dense(32))
# model.add(Dropout(0.3))
# model.add(Dense(16))
# model.add(Dropout(0.3))
# model.add(Dense(16))
# model.add(Dropout(0.3))
# model.add(Dense(1))

# model.summary()

# # input1 = Input(shape=(10))
# # dense1 = Dense(128)(input1)
# # drop1 = Dropout(0.3)(dense1)
# # dense2 = Dense(128)(drop1)
# # drop2 = Dropout(0.3)(dense2)
# # dense3 = Dense(128)(drop2)
# # drop3 = Dropout(0.3)(dense3)
# # dense4 = Dense(64)(drop3)
# # drop4 = Dropout(0.3)(dense4)
# # dense5 = Dense(64)(drop4)
# # drop5 = Dropout(0.3)(dense5)
# # dense6 = Dense(64)(drop5)
# # drop6 = Dropout(0.3)(dense6)
# # dense7 = Dense(64)(drop6)
# # drop7 = Dropout(0.3)(dense7)
# # dense8 = Dense(32)(drop7)
# # drop8 = Dropout(0.3)(dense8)
# # dense9 = Dense(32)(drop8)
# # drop9 = Dropout(0.3)(dense9)
# # dense10 = Dense(32)(drop9)
# # drop10 = Dropout(0.3)(dense10)
# # dense11 = Dense(16)(drop10)
# # drop11 = Dropout(0.3)(dense11)
# # dense12 = Dense(16)(drop11)
# # drop12 = Dropout(0.3)(dense12)
# # output1 = Dense(1)(drop12)
# # model = Model(inputs = input1, outputs = output1)

# # model.summary()

model = Sequential()
model.add(Conv2D(32, 2, activation='relu', input_shape=(5,2,1), padding='same'))
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
model.add(Dense(1, activation='sigmoid'))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()

es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience = 30,
    restore_best_weights=True
)
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path1 = './_save/keras39/07_dacon_diabetes/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # '1000-0.7777.hdf5'  #fit에서 반환되는 값을 빼오는 것이다. 
filepath = "".join([path1, 'k30_', date, '_', filename])

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose = 1,
    save_best_only=True,
    filepath = filepath
)



hist = model.fit(x_train, y_train, epochs=1000, batch_size=2, verbose=1, 
                 validation_split=0.25, callbacks=[es, mcp])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

print("로스 : ", loss)
print('r2 스코어 :',r2 )
print('시간 : ', round(end-start, 3), '초')


# print("r2스코어 : ", r2)
# print("걸린시간 : ", round(end - start, 2), "초" )
# print('=====================hist==========')
# print(hist)
# print('======================= hist.history==================')
# print(hist.history)
# print('================loss=================')
# print(hist.history['loss'])
# print('=================val_loss==============')
# print(hist.history['val_loss'])
# print('====================================================')

# import matplotlib.pyplot as plt
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c='red', label='loss')
# plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
# plt.legend(loc='upper right') #라벨 값이 무엇인지 명시해주는 것이 레전드
# plt.title('다이어베츠 Loss') #그래프의 제목 
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid()
# plt.show()

# 로스 :  3184.015869140625
# r2 스코어 : 0.4863319871299602

# minmaxscaler
# 로스 :  3053.407958984375
# r2 스코어 : 0.5074025892828946

# StandardScaler
# 로스 :  2953.283203125
# r2 스코어 : 0.5235553960095585

# maxabsscaler
# 로스 :  3072.412841796875
# r2 스코어 : 0.5043365521694105

# RobustScaler

# 세이브 점수
# 로스 :  3196.185791015625
# r2 스코어 : 0.48436862012275794

# drop out
# 로스 :  3207.538330078125
# r2 스코어 : 0.48253717220876147

# cpu
# 로스 :  3100.33544921875
# r2 스코어 : 0.4998319406804721
# 시간 :  5.361 초

# gpu
# 로스 :  2982.049072265625
# r2 스코어 : 0.5189147019050072
# 시간 :  38.105 초

# 로스 :  27927.12109375
# r2 스코어 : -3.5054010328549046
# 시간 :  14.321 초