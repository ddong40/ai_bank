import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from tensorflow.keras.utils import to_categorical
import time

#1 데이터

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

print(np.unique(y_train, return_counts=True))

x_train = x_train/255.
x_test = x_test/255.

ohe = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

#2 모델 구성
model = Sequential()
model.add(Conv2D(filters = 100, kernel_size=(3,3), activation='relu', input_shape = (32,32,3)))
model.add(Conv2D(50,(3,3), activation='relu'))
model.add(Conv2D(20,(2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dense(100, activation = 'softmax'))

#3 컴파일 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience=300,
    verbose=1,
    restore_best_weights=True
)
model.fit(x_train, y_train, epochs=10000, batch_size=100, verbose=1, validation_split=0.25, callbacks=[es])

# 평가예측
loss= model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

print('로스 값 : ', loss[0])
print('y_predict[9999] 값 : ', y_predict[9999])

