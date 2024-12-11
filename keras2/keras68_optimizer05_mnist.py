
import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import tensorflow as tf
import random as rn 
rn.seed(337)
tf.random.set_seed(337)
np.random.seed(337)

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data() #알아서 데이터 나눠줌
# print(x_train)

##### 스케일링 1-1
x_train = x_train/255.
x_test = x_test/255.

x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

print(x_train.shape, x_test.shape)

##원핫 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

##원핫 1-1
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(sparse=False)
# y_train = y_train.reshape(-1, 1)
# y_test = y_test.reshape(-1, 1)
# y_train = ohe.fit_transform(y_train)
# y_test = ohe.transform(y_test)

###원핫 1-2
# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)

# y_test = y_test.to_numpy()

# print(np.max(x_train), np.min(x_train)) #1.0 0.0
# print(y_train.shape)


###### 스케일링 1-2 
# x_train = (x_train - 127.5) / 127.5
# x_test = (x_test - 127.5) / 127.5
# print(np.max(x_test), np.min(x_test)) # 1.0 -1.0

# ##### 스케일링 2. MinMaxScaler(), StandardScaler
# x_train = x_train.reshape(60000, 28*28)
# x_test = x_test.reshape(10000, 28*28)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.max(x_train), np.min(x_train))

#2 모델
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(28*28,)))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

#3 컴파일 훈련

from tensorflow.keras.optimizers import Adam

lr = np.array([0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001])

for i in lr:

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=i), metrics=['accuracy'])

    start_time = time.time()

    es = EarlyStopping(
        monitor='val_loss',
        mode= 'min',
        verbose=1,
        patience=30,
        restore_best_weights=True
    )

    import datetime 
    date = datetime.datetime.now() #데이트라는 변수에 현재 시간을 반환한다.
    print(date) #2024-07-26 16:49:51.174797
    print(type(date)) #<class 'datetime.datetime'>
    date = date.strftime("%m%d_%H%M") #시간을 문자열로 바꿔줌
    print(date) #0726_1654
    print(type(date))


    path = './_save/keras38/_mnist/'
    filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # '1000-0.7777.hdf5'  #fit에서 반환되는 값을 빼오는 것이다. 
    filepath = "".join([path, 'k35_04', date, '_', filename])


    mcp = ModelCheckpoint(
        monitor='val_loss',
        mode = 'auto',
        verbose=1,
        save_best_only=True,
        filepath = filepath
    )

    model.fit(x_train, y_train, epochs=1, batch_size=10, verbose=1, validation_split=0.25, callbacks=[es, mcp])

    end_time = time.time()

    #4. 예측 평가
    loss = model.evaluate(x_test, y_test)
    y_predict = model.predict(x_test)

    # y_test = np.argmax(y_test).reshape(-1,1)
    # y_predict = np.argmax(y_predict).reshape(-1,1)

    # acc = accuracy_score(y_test, y_predict)

    print('{0} : '.format(i))
    print("로스값 : ", loss[0])
    print("정확도 : ", loss[1])
    print('----------------------------')

# padding
# 로스 :  [0.2835237383842468, 0.9211999773979187]
# acc_score : 0.9212

# dnn으로 변환
# 로스 :  0.1174471452832222
# 정확도 : 0.9758
# 시간  1073.563 초