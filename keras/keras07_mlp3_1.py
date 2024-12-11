import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)])
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [10,9,8,7,6,5,4,3,2,1]])
x=x.T
y=np.transpose(y)

print(x.shape) #(10, 3)
print(y.shape) #(10, 2)

#2.모델
# [실습] 맹그러봐
# x_predict = [10, 31, 211]
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(2))

#3 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=250, batch_size=1)

#4 평가 예측
loss = model.evaluate(x, y)
result = model.predict([[10, 31, 211]])
print("loss값 : ", loss)
print("예측 값 : ", result)

#loss값 :  0.05420089513063431
#예측 값 :  [[10.353589   -0.17548297]]
#epochs=250, batch_size=1