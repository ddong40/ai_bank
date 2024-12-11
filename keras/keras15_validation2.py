import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

x_train = x[:13]
y_train = y[:13]

x_val = x[9:14]
y_val = y[9:14]

x_test = x[11:]
y_test = y[11:]

print(x_val)
print(x_test)

#2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs= 100, batch_size=10, verbose=1, validation_data = (x_val, y_val))

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
result = model.predict([17])
print("loss값 : ", loss)
print("예측 값 : ", result)