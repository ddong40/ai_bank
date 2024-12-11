#accuracy 95프로 만들기
#cifar10 뭔지 찾아보기

import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

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

data_gen = ImageDataGenerator(
    rotation_range=0.1,
    width_shift_range = 0.1,
    horizontal_flip= True,
    vertical_flip= True,
    fill_mode = 'nearest'
)

augment_size = 50000

randidx = np.random.randint(x_train.shape[0], size = augment_size)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

print(x_augmented.shape)

x_augmented = data_gen.flow(
    x_augmented, y_augmented,
    batch_size= augment_size,
    shuffle=True,
    save_to_dir='C:/Users/ddong40/ai_2/_data/_save_img/03_cifar10/'
).next()[0]

# print(y_augmented.shape)
'''
x_train = x_train.reshape(-1, 32, 32, 3)
x_test = x_test.reshape(-1, 32, 32, 3)

x_train = np.concatenate((x_train, x_augmented), axis = 0)
y_train = np.concatenate((y_train, y_augmented), axis = 0)

print(x_train.shape)
print(y_train.shape)


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

print(y_train.shape)

# import matplotlib.pyplot as plt
# plt.imshow(x_train[49999]) #xtrain의 59999번째를 보여주겠다 , 'gray' 색상을 흑백으로 하겠다. 
# plt.show()
# print('y_train[49999]의 값 : ', y_train[49999])

#2. 모델 구성
model = Sequential()
input1 = Input(shape = (32, 32, 3))
conv2d1 = Conv2D(64, 2, strides=1, padding='same')(input1)
maxpool = MaxPooling2D()(conv2d1)
drop1 = Dropout(0.2)(maxpool)
conv2d2 = Conv2D(32, 2, activation= 'relu', strides=1, padding='same')(drop1)
drop2 = Dropout(0.2)(conv2d2)
conv2d3 = Conv2D(32, 2, activation= 'relu', strides=1, padding='same')(drop2)
drop3 = Dropout(0.2)(conv2d3)
conv2d4 = Conv2D(32, 2, activation= 'relu', strides=1, padding='same')(drop3)
flatten = Flatten()(conv2d4)
dense1 = Dense(16, activation='relu')(flatten)
output1 = Dense(10, activation = 'softmax')(dense1)
model = Model(inputs=input1, outputs=output1)


# model = Sequential()
# model.add(Conv2D(64, (2,2), input_shape = (32, 32, 3),
#                  strides=1,
#                  padding='same'))
# model.add(MaxPooling2D())
# model.add(Dropout(0.2))
# model.add(Conv2D(filters=32, kernel_size=(2,2), activation= 'relu',
#                  strides=1,
#                  padding='same'))
# model.add(Dropout(0.2))
# model.add(Conv2D(filters=32, kernel_size=(2,2), activation= 'relu',
#                  strides=1,
#                  padding='same'))
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (2,2), activation= 'relu',
#                  strides=1,
#                  padding='same'))
# model.add(Flatten())
# model.add(Dense(units=16, input_shape = (32, ), activation='relu'))
# model.add(Dense(10, activation='softmax'))

#3. 컴파일 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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

path1 = './_save/keras49/cifa10/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path1, 'k30_', date, '_', filename])


mcp = ModelCheckpoint(
    monitor= 'val_loss',
    mode = 'auto',
    verbose=1,
    save_best_only= True,
    filepath = filepath
)


model.fit(x_train, y_train, epochs = 100, batch_size = 128, verbose=1, validation_split=0.25, callbacks=[es, mcp])

#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

y_test = np.argmax(y_test, axis=1).reshape(-1, 1)
y_predict = np.argmax(y_predict, axis=1).reshape(-1, 1)

acc = accuracy_score(y_test, y_predict)

print('로스 : ', loss[0])
print('acc : ', acc)

# padding
# 로스 :  1.2356504201889038
# acc :  0.558

# Maxpooling2D
# 로스 :  0.9633400440216064
# acc :  0.6715

# 함수형
# 로스 :  0.9597545862197876
# acc :  0.6775

# 로스 :  1.787427544593811
# acc :  0.3614
'''