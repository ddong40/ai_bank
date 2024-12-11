from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일훈련
epochs = 6200
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=epochs)

#4. 평가, 예측

loss = model.evaluate(x, y)
print("=======================================")
print("epochs : ", epochs)
print("로스 : ", loss)
result = model.predict([6])
print("6의 예측값 :", result)

#0.41720420122146606 / 2400
#0.38000041246414185 / 2700

# epochs :  5450
# 로스 :  0.3799999952316284
# 6의 예측값 : [[5.700001]]

