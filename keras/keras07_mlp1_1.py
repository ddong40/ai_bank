import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
# x = np.array([[1,2,3,4,5],
            #  [6,7,8,9,10]])
x = np.array([[1,6], [2,7], [3,8], [4,9],[ 5,10]])
y = np.array([1,2,3,4,5])

print(x.shape) # (5,2)
print(y.shape) # (5,)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=2)) #column의 개수만큼 input_dim을 넣어준다.
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가 예측
loss = model.evaluate(x, y)
result = model.predict([[6, 11]]) #
print('로스 : ', loss)
print('[6, 11]의 예측값 :', result)

#[실습] : 소수 2째짜리까지 맞춰