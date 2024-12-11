import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

#train_test_split으로만 3등분으로 잘라라

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, shuffle= False, random_state= 100 )
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size= 0.4, shuffle= False, random_state= 100 )

print(x_test)
print(x_val)
print(x_train)

#2. 모델구성

model = Sequential()
model.add(Dense(1, input_dim = 1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3 컴파일 훈련

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs= 10, batch_size= 10, verbose=1, validation_data=(x_val, y_val))

#4 평가예측

loss = model.evaluate()