import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10)])
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [10,9,8,7,6,5,4,3,2,1],
              [9,8,7,6,5,4,3,2,1,0]])
x=x.T
y=np.transpose(y)

print(x.shape) #(10, 1)
print(y.shape) #(10, 3)

#2 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(3))

#3 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=2)

#4 평가 예측
loss = model.evaluate(x, y)
result = model.predict([10])
print("loss 값 : ", loss)
print("result : ", result)


# loss 값 :  0.0005763366934843361
# result :  [[10.99685    -0.03320704 -0.9611916 ]]