import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
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

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

ohe = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

#2 모델 구성
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(3072,)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(100, activation = 'softmax'))

#3 컴파일 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

start_time = time.time()

import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path1 = './_save/keras38/_cifar100/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path1, 'k30_', date, '_', filename])

es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience=30,
    verbose=1,
    restore_best_weights=True
)

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only = True,
    verbose=1,
    filepath=filepath
)

model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=1, validation_split=0.25, callbacks=[es, mcp])
end_time= time.time()

# 평가예측
loss= model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1).reshape(-1, 1) 
y_test = np.argmax(y_test, axis=1).reshape(-1, 1)

acc = accuracy_score(y_test, y_predict)

print('로스 값 : ', loss[0])
print('acc 스코어 : ', round(loss[1], 3))
print('시간 ', round(end_time - start_time, 3), '초')

# 로스 값 :  2.7884061336517334
# acc 스코어 :  0.318
# y_predict[9999] 값 :  [62]

# padding 했을 때
# 로스 값 :  2.8896849155426025
# acc 스코어 :  0.301

# 로스 값 :  2.870060443878174
# acc 스코어 :  0.302

# Maxpooling2D
# 로스 값 :  2.6650283336639404
# acc 스코어 :  0.341

# dnn변환
# 로스 값 :  4.267077922821045
# acc 스코어 :  0.039
# 시간  220.573 초