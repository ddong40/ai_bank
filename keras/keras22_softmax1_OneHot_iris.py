# onehot 사이킷런, 케라스, 판다스에 있음 찾으셈
# 


import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import time

#1. 데이터
datasets = load_iris()
# print(datasets)
print(datasets.DESCR)
print(datasets.feature_names)


x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (150, 4) (150,)

print(y)

print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([50, 50, 50], dtype=int64))
print(pd.value_counts(y))
# 0    50
# 1    50
# 2    50

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, shuffle=True, 
                                                    random_state= 100, stratify=y)

print(pd.value_counts(y_train))

#데이터 reshpae할 때 조건

# 1.데이터의 내용이 바뀌면 안된다.
# 2.데이터의 순서가 바뀌면 안된다. 

# from tensorflow.keras.utils import to_categorical #케라스
# y = to_categorical(y)

# from sklearn.preprocessing import OneHotEncoder #사이킷런 # preprocessing은 한국말로 전처리
# ohe = OneHotEncoder(sparse=False)
# y = pd.DataFrame(y)
# y = ohe.fit_transform(y)

# y = pd.get_dummies(y) #pandas

# y = pd.DataFrame(y)
# y = pd.get_dummies(y[0]) 
# print(y)


#사이킷런 선생님꺼 
from sklearn.preprocessing import OneHotEncoder
y_ohe3 = y.reshape(-1,1)
ohe = OneHotEncoder(sparse=False) #True가 디폴트
y_ohe3 = ohe.fit_transform(y_ohe3)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, shuffle=True, random_state= 100)


#2 모델 구성
model = Sequential()
model.add(Dense(128, activation = 'relu', input_dim=4))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(3, activation= 'softmax'))

#3 컴파일 훈련
from sklearn.metrics import accuracy_score

# model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
start_time = time.time()
es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience= 60,
    restore_best_weights = True
    )

model.fit(x_train, y_train, epochs= 1000, batch_size=10, verbose=1, validation_split=0.25, callbacks=[es])
end_time = time.time()
#4 평가 예측
loss = model.evaluate(x_test, y_test)
print("로스 :", loss[0])
print("acc : ", round(loss[1], 3))
print("걸린시간 : ", round(end_time - start_time, 2), "초" )
y_pred = model.predict(x_test)
y_pred = np.round(y_pred)

print(y_pred)

#로스 : 0.017849013209342957
# acc :  1.0
# random_state를 100으로 주었을 때 batch를 10으로 주었을 때 patience 60   
