import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_california_housing
import sklearn as sk


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x)
print(y)  # [4.526 3.585 3.521 ... 0.923 0.847 0.894] 벡터 1개
print(x.shape, y.shape)   # (20640, 8) (20640,)  인풋 8 ,아웃풋 1

# [실습] 맹그러
# R2 0.59 이상 -0.1 (0.49)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split (x, y,
                                                     train_size=0.7,
                                                     shuffle=True,
                                                     random_state=104) 

print(x)
print(x.shape)
print(y)
print(y.shape)


#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=8))
model.add(Dense(30))
model.add(Dense(33))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=100, batch_size=11, verbose = 2, validation_split = 0.3)

#4. 평가, 예측
print("+++++++++++++++++++++++++++++++++++++++++")
loss = model.evaluate(x_test, y_test)
print("로스 : ", loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)
print("r2스코어 : ", r2)

print(hist)
print('=======================hist.history=======================')
print(hist.history)
print('=======================loss==========================')
print(hist.history['loss'])
print('=======================val_loss=======================')
print(hist.history['val_loss'])
print('=======================================================')

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label = 'val_loss')
plt.legend(loc='upper right')
plt.title('california Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()









# 로스 :  0.6514161825180054
# r2스코어 :  0.5194052880427066

# 로스 :  0.6381560564041138
# r2스코어 :  0.5291882119746794

# random_state=104
# model.add(Dense(1, input_dim=8))
# model.add(Dense(30))
# model.add(Dense(33))
# model.add(Dense(1))
# 로스 :  0.6738346815109253
# r2스코어 :  0.513691900770961