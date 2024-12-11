from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6]) #데이터를 추가했다. 
y = np.array([1,2,3,4,5,6])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=360)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("로스 : ", loss) #로스를 추가했다. 
result = model.predict([1,2,3,4,5,6,7]) #예측 데이터를 추가했다. 
print("7의 예측값 :", result)