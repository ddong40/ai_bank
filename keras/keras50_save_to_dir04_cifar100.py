import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

datagen = ImageDataGenerator(
    rotation_range=0.1,
    width_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

augment_size = 50000

randidx = np.random.randint(x_train.shape[0], size = augment_size)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

print(x_augmented.shape)

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2],3)

x_augmented = datagen.flow(
    x_augmented, y_augmented,
    batch_size= augment_size,
    shuffle = False,
    save_to_dir= 'C:/Users/ddong40/ai_2/_data/_save_img/04_cifar100/'
).next()[0]

'''
x_train = x_train.reshape(50000, 32,32,3)
x_test = x_test.reshape(10000, 32,32,3)

x_train = np.concatenate((x_train, x_augmented), axis=0)
y_train = np.concatenate((y_train, y_augmented), axis=0)


ohe = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

#2 모델 구성
model = Sequential()
input1 = Input(shape=(32, 32, 3))
conv2d1 = Conv2D(32, 2, activation='relu', strides=1, padding='same')(input1)
maxpool = MaxPooling2D()(conv2d1)
drop1 = Dropout(0.2)(maxpool)
conv2d2 = Conv2D(32, 2, activation='relu', strides=1, padding='same')(drop1)
conv2d3 = Conv2D(32, 2, activation='relu', strides=1, padding='same')(conv2d2)
conv2d4 = Conv2D(32, 2, activation='relu', strides=1, padding='same')(conv2d3)
flatten = Flatten()(conv2d4)
dense1 = Dense(128)(flatten)
drop2 = Dropout(0.2)(dense1)
output1 = (Dense(100, activation = 'softmax'))(drop2)
model = Model(inputs = input1, outputs = output1)


# model = Sequential()
# model.add(Conv2D(filters = 32, kernel_size=(2,2), activation='relu', input_shape = (32,32,3),
#                  strides=1,
#                  padding='same'))
# model.add(MaxPooling2D())
# model.add(Dropout(0.2))
# model.add(Conv2D(32,(2,2), activation='relu',
#                  strides=1,
#                  padding='same'))
# model.add(Conv2D(32,(2,2), activation='relu',
#                  strides=1,
#                  padding='same'))
# model.add(Conv2D(32,(2,2), activation='relu',
#                  strides=1,
#                  padding='same'))
# model.add(Flatten())
# model.add(Dense(128, input_shape= (32,)))
# model.add(Dropout(0.2))
# model.add(Dense(100, activation = 'softmax'))

#3 컴파일 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path1 = './_save/keras49/cifar100/'
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

# 평가예측
loss= model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1).reshape(-1, 1) 
y_test = np.argmax(y_test, axis=1).reshape(-1, 1)

acc = accuracy_score(y_test, y_predict)

print('로스 값 : ', loss[0])
print('acc 스코어 : ', round(loss[1], 3))
print('y_predict[9999] 값 : ', y_predict[9999])

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
'''