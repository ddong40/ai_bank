import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import sklearn as sk
from sklearn.datasets import load_diabetes
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
start_time = time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.3)
end_time = time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("로스 : ", loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.legend(loc='upper right') #라벨 값이 무엇인지 명시해주는 것이 레전드
plt.title('kaggle Loss') #그래프의 제목 
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()