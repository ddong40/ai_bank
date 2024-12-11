#35_2에서 가져옴
# x_train, x_test는 reshape
# y_tset, y_train OneHotEncoding

import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from sklearn.metrics import accuracy_score

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data() #알아서 데이터 나눠줌
# print(x_train)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
 

print(x_train[0])
print("y_train[0] : ", y_train[0])

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,) #색상 값이 숨겨져있다. 사실은 60000, 28, 28, 1
print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,) 

#2 모델구성

model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(28, 28, 1))) #데이터는 4차원, input_shape는 3차원 행을 빼주고
                            # shape = (batch_size, rows, columns, channels)
                            # shape = (batch_size, heights, widths, channels)
#batch_size인 이유는? 데이터를 몇개씩 넣어서 훈련시키는 것이 모델의 구조를 바꾸진 않기 때문에 훈련시킬 양을 조절 

model.add(Conv2D(filters=20, kernel_size=(3,3))) #공식적인 명칭 filter가 증폭시킬 량, kernel size가 이미지를 쪼갤 사이즈의 비율
# 즉 28, 28, 1 의 이미지가 28, 28, 20의 필터가 된다. 
model.add(Conv2D(15, (4,4)))
model.add(Flatten()) #평탄화 했으니 

model.add(Dense(units=8)) #dense는 결과가 2차원이지만 2차원 이상도 먹힘
model.add(Dense(units=9, input_shape=(8,))) #4차원을 2차원으로 변환하는 작업이 필요하다. #순서와 값이 바뀌지 않는 한에서 reshape한다. 
 #shape = (batch_size, input_dim) 
model.add(Dense(10, activation='softmax')) #y는 60000,10 으로 onehot encoding해야한다




model.summary() #summary가 됨. 

'''
#3 컴파일, 훈련
model.compile( loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs=2, batch_size=128)

#4 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('로스 값', loss)
print('정확도 ', acc)
'''