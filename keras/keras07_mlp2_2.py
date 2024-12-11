import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1 데이터
x = np.array(range(10)) 
print(x) # [0 1 2 3 4 5 6 7 8 9]
print(x.shape) #(10,)

x = np.array(range(1,10))
print(x) # [1 2 3 4 5 6 7 8 9]
print(x.shape)

x = np.array([range(10), range(21,31), range(201,211)])
print(x)
print(x.shape)
x = x.T
print(x)
print(x.shape) # (10, 3)

y = np.array([1,2,3,4,5,6,7,8,9,10])

#[실습]
#[10 ,31, 211] 예측할 것

#2 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=5)

#4 평가 예측
loss = model.evaluate(x, y)
result = model.predict(np.array([[10, 31, 211]]))
print("loss 값 : ",loss)
print("예측 값 : ", result)

# epochs = 500, batch_size=5
# loss 값 :  3.991170949291245e-09
# 예측 값 :  [[10.999864]]