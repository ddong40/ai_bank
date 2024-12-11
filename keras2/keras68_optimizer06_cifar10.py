#accuracy 95프로 만들기
#cifar10 뭔지 찾아보기

import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import time
import tensorflow as tf
import random as rn 
rn.seed(337)
tf.random.set_seed(337)
np.random.seed(337)

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data() 

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
# (50000, 32, 32, 3)
# (10000, 32, 32, 3)
# (50000, 1)
# (10000, 1)

x_train = x_train/255.
x_test = x_test/255.

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

# x_train = x_train.reshape(50000, 32, 23, 3)
# x_test = x_test.reshape()

# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)

# y_train = y_train.to_numpy()
# y_test = y_test.to_numpy()

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

# import matplotlib.pyplot as plt
# plt.imshow(x_train[49999]) #xtrain의 59999번째를 보여주겠다 , 'gray' 색상을 흑백으로 하겠다. 
# plt.show()
# print('y_train[49999]의 값 : ', y_train[49999])

#2. 모델 구성
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
model.add(Dense(10, activation='softmax'))

from tensorflow.keras.optimizers import Adam

lr = np.array([0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001])

for i in lr : 

    #3. 컴파일 훈련
    model.compile(loss = 'categorical_crossentropy', optimizer=Adam(learning_rate=i), metrics=['accuracy'])

    start_time = time.time()

    es = EarlyStopping(
        monitor = 'val_loss',
        mode = 'min',
        verbose=1,
        patience=20,
        restore_best_weights=True
    )

    import datetime
    date = datetime.datetime.now()
    date = date.strftime('%m%d_%H%M')

    path1 = './_save/keras38/_cifa10/'
    filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
    filepath = ''.join([path1, 'k30_', date, '_', filename])


    mcp = ModelCheckpoint(
        monitor= 'val_loss',
        mode = 'auto',
        verbose=1,
        save_best_only= True,
        filepath = filepath
    )

    model.fit(x_train, y_train, epochs = 1, batch_size = 128, verbose=1, validation_split=0.25, callbacks=[es, mcp])
    end_time = time.time()

    #평가 예측
    loss = model.evaluate(x_test, y_test)
    y_predict = model.predict(x_test)

    # y_test = np.argmax(y_test, axis=1).reshape(-1, 1)
    # y_predict = np.argmax(y_predict, axis=1).reshape(-1, 1)

    # acc = accuracy_score(y_test, y_predict)
    print('{0} : '.format(i))
    print("로스값 : ", loss[0])
    print("정확도 : ", loss[1])
    print('----------------------------')
 

# padding
# 로스 :  1.2356504201889038
# acc :  0.558

# Maxpooling2D
# 로스 :  0.9633400440216064
# acc :  0.6715


# 로스 :  1.8835355043411255
# acc :  0.2936
# 시간 :  35.547 초

# 증폭
# 로스 :  1.787427544593811
# acc :  0.3614