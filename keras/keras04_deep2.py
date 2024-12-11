from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

# [실습] 레이어의 깊이와 노드의 갯수를 이용해서 최소의 loss를 맹그러
# 에포는 100으로 고정, 건들지 말것!
# 로스 기준 0.33 이하!!!


#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))



#3. 컴파일훈련
epochs = 100
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=epochs)

#4. 평가, 예측

loss = model.evaluate(x, y)
print("=======================================")
print("epochs : ", epochs)
print("로스 : ", loss)
result = model.predict([6])
print("6의 예측값 :", result)


