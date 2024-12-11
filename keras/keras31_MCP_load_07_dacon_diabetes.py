import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import sklearn as sk
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import EarlyStopping
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


# print(x)
# print(y)
# print(x.shape, y.shape)   # (442, 10) (442,)
# 분류 데이터는 0과 1만 있음 y값이 종류가 많으면 폐기모델

# [실습]
# R2 0.62 이상 -0.1 (0.52)

#2. 모델구성
# model = Sequential()
# model.add(Dense(128, input_dim=10))
# model.add(Dense(128))
# model.add(Dense(128))
# model.add(Dense(64))
# model.add(Dense(64))
# model.add(Dense(64))
# model.add(Dense(64))
# model.add(Dense(32))
# model.add(Dense(32))
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(16))
# model.add(Dense(1))

# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
# start = time.time()

# es = EarlyStopping(
#     monitor='val_loss',
#     mode = 'min',
#     patience = 30,
#     restore_best_weights=True
# )

# hist = model.fit(x_train, y_train, epochs=1000, batch_size=2, verbose=1, 
#                  validation_split=0.25, callbacks=[es])
# end = time.time()

#4. 평가, 예측
model = load_model('./_save/keras30_mcp/07_dacon_diabetes/k30_0726_2050_0018-2720.9097.hdf5')
loss = model.evaluate(x_test, y_test)


y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

print("로스 : ", loss)
print('r2 스코어 :',r2 )

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

# load data
# 로스 :  3196.185791015625
# r2 스코어 : 0.48436862012275794