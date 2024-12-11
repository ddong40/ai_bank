from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

# [실습] 레이어의 깊이와 노드의 갯수를 이용해서 [6]을 맹그러
# 에포는 100으로 고정, 건들지 말것!
# 소수 네째자리까지 맞추면 합격, 예 : 6.0000 또는 5.9999


#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(4, input_dim=3))
model.add(Dense(7, input_dim=4))
model.add(Dense(10, input_dim=7))
model.add(Dense(50, input_dim=10))
model.add(Dense(100, input_dim=50))
model.add(Dense(300, input_dim=100))
model.add(Dense(500, input_dim=300))
model.add(Dense(700, input_dim=500))
model.add(Dense(1000, input_dim=700))
model.add(Dense(1500, input_dim=1000))
model.add(Dense(900, input_dim=1500))
model.add(Dense(2000, input_dim=900))
model.add(Dense(2000, input_dim=2000))
model.add(Dense(2000, input_dim=2000))
model.add(Dense(2000, input_dim=2000))
model.add(Dense(2000, input_dim=2000))
model.add(Dense(2000, input_dim=2000))
model.add(Dense(2000, input_dim=2000))
model.add(Dense(2000, input_dim=2000))
model.add(Dense(2000, input_dim=2000))
model.add(Dense(2000, input_dim=2000))
model.add(Dense(2000, input_dim=2000))
model.add(Dense(2000, input_dim=2000))
model.add(Dense(2000, input_dim=2000))
model.add(Dense(2000, input_dim=2000))
model.add(Dense(2000, input_dim=2000))
model.add(Dense(2000, input_dim=2000))
model.add(Dense(2000, input_dim=2000))
model.add(Dense(2000, input_dim=2000))
model.add(Dense(2000, input_dim=2000))
model.add(Dense(2000, input_dim=2000))
model.add(Dense(2000, input_dim=2000))
model.add(Dense(2000, input_dim=2000))
model.add(Dense(2000, input_dim=2000))
model.add(Dense(1000, input_dim=2000))
model.add(Dense(900, input_dim=1000))
model.add(Dense(700, input_dim=900))
model.add(Dense(450, input_dim=700))
model.add(Dense(400, input_dim=450))
model.add(Dense(300, input_dim=400))
model.add(Dense(200, input_dim=300))
model.add(Dense(100, input_dim=200))
model.add(Dense(3, input_dim=100))
model.add(Dense(1, input_dim=3))



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

