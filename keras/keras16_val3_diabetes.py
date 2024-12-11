import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import sklearn as sk

from sklearn.datasets import load_diabetes

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=250)


print(x)
print(y)
print(x.shape, y.shape)   # (442, 10) (442,)
# 분류 데이터는 0과 1만 있음 y값이 종류가 많으면 폐기모델

# [실습]
# R2 0.62 이상 -0.1 (0.52)

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=10))
model.add(Dense(70))
model.add(Dense(69))
model.add(Dense(49))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.3)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("로스 : ", loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)


# random_state=250
# 로스 :  3018.037841796875
# r2스코어 :  0.5131087413055728

# random_state=250
# model.add(Dense(70))
# model.add(Dense(64))
# model.add(Dense(49))
# 로스 :  2980.6044921875
# r2스코어 :  0.5191477931431967