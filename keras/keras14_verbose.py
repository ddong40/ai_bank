# 08 -1 카피

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
# x = np,array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10])
x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])

x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=1,
          verbose = 2
          )

# verbose=0 : 침묵
# verbose=1 : 디폴트
# verbose=2 : 프로그래스바 삭제
# vwrbose=나머지 : epoch만 나옴

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("++++++++++++++++++++++++++++++++++")
results = model.predict([11]) #
print("로스: ", loss)
print('[11]의 예측값 : ', results)