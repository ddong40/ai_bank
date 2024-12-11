import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_wine
import time

#1 데이터
datasets = load_wine()

x = datasets.data
y = datasets.target

print(datasets)
print(datasets.DESCR)

from sklearn.preprocessing import OneHotEncoder
y = y.reshape(-1,1)
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, shuffle= True, 
                                                    random_state= 150, stratify=y)


#2 모델구성

model = Sequential()
model.add(Dense(128, 'relu', input_dim=13))
model.add(Dense(128, 'relu'))
model.add(Dense(128, 'relu'))
model.add(Dense(128, 'relu'))
model.add(Dense(128, 'relu'))
model.add(Dense(64, 'relu'))
model.add(Dense(64, 'relu'))
model.add(Dense(64, 'relu'))
model.add(Dense(64, 'relu'))
model.add(Dense(64, 'relu'))
model.add(Dense(3, 'softmax'))

#컴파일 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
start_time = time.time()
es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 50,
    restore_best_weights= True)
model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=1, validation_split=0.25, callbacks=[es])
end_time = time.time()

#평가 예측
loss = model.evaluate(x_test, y_test)
print("로스값 : ", loss[0])
print("accuracy : ", round(loss[1],3))
print("걸린시간 : ", round(end_time - start_time, 2), "초" )
y_pred = model.predict(x_test)
print(y_pred)
