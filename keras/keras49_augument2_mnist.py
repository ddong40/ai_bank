#35_2에서 가져옴
# x_train, x_test는 reshape
# y_tset, y_train OneHotEncoding

import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data() #알아서 데이터 나눠줌
# print(x_train)

##### 스케일링 1-1
x_train = x_train/255.
x_test = x_test/255.

# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)

# print(x_train.shape)

###원핫 
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

##원핫 1-1
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(sparse=False)
# y_train = y_train.reshape(-1, 1)
# y_test = y_test.reshape(-1, 1)
# y_train = ohe.fit_transform(y_train)
# y_test = ohe.transform(y_test)

train_datagen = ImageDataGenerator(
    horizontal_flip = False,
    vertical_flip = False,
    width_shift_range = 0.1,
    rotation_range= 0.1,
    fill_mode = 'nearest'    
)

augment_size = 40000

randidx = np.random.randint(x_train.shape[0], size=augment_size )

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

print(x_augmented.shape)

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2], 1)

print(x_augmented.shape)
print(y_augmented.shape)

x_augmented = train_datagen.flow(
    x_augmented, y_augmented, 
    batch_size = augment_size,
    shuffle=False
).next()[0]

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

x_augmented = np.concatenate((x_train, x_augmented), axis=0)
y_augmented = np.concatenate((y_train, y_augmented), axis=0)

print(x_train.shape)
print(x_test.shape)
print(x_augmented.shape)
print(y_augmented.shape)


###원핫 1-2
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

y_test = y_test.to_numpy()

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




# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)
# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)
 

# print(x_train[0])
# print("y_train[0] : ", y_train[0])

# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,) #색상 값이 숨겨져있다. 사실은 60000, 28, 28, 1
# print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,) 


#2 모델구성

# input1 = Input(shape=(13))
# dense1 = Dense(64, name='ys1')(input1)
# drop1 = Dropout(0.3)(dense1)
# dense2 = Dense(64, name='ys2', activation = 'relu')(drop1)

model = Sequential()
# input1 = Input(shape=(28, 28, 1))
# conv2d1 = Conv2D(64, 2, name='ys1', strides=1, padding='same')(input1)
# maxpool = MaxPooling2D()(conv2d1)
# conv2d2 = Conv2D(64, 2, name='ys2', strides=1, padding='same')(maxpool)
# drop1 = Dropout(0.2)(conv2d2)
# conv2d3 = Conv2D(32, 2, name='ys3', strides=1, padding='same')(drop1)
# flatten = Flatten()(conv2d3)
# dense1 = Dense(32, name='ys4')(flatten)
# drop2 = Dropout(0.2)(dense1)
# dense2 = Dense(16, name='ys5')(drop2)
# output1 = Dense(10, activation='softmax')(dense2)
# model = Model(inputs=input1, outputs=output1)

model.add(Conv2D(64, (2,2), input_shape=(28, 28, 1), 
                 strides=1,
                 padding='same')) 
model.add(MaxPooling2D())
model.add(Conv2D(filters=64, kernel_size=(2,2),
                 strides=1,
                 padding='same')) 
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2),
                 strides=1,
                 padding='same'))
model.add(Flatten()) 
model.add(Dense(units=32))
model.add(Dropout(0.2)) 
model.add(Dense(units=16, input_shape=(32,))) 
model.add(Dense(10, activation='softmax')) #y는 60000,10 으로 onehot encoding해야한다




model.summary() #summary가 됨. 


#3 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 10,
    verbose = 1,
    restore_best_weights=True
)

#################mcp 세이브 파일명 만들기 시작##################

import datetime 
date = datetime.datetime.now() #데이트라는 변수에 현재 시간을 반환한다.
print(date) #2024-07-26 16:49:51.174797
print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M") #시간을 문자열로 바꿔줌
print(date) #0726_1654
print(type(date))


path = './_save/keras49/mnist/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # '1000-0.7777.hdf5'  #fit에서 반환되는 값을 빼오는 것이다. 
filepath = "".join([path, 'k35_04', date, '_', filename])

############mcp세이브 파일명 만들기 끝################

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose = 1, #가장 좋은 지점을 알려줄 수 있게 출력함
    save_best_only=True,
    filepath = filepath
)
# 생성 예" './_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5'


start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=1, validation_split = 0.25, callbacks=[es, mcp])
end = time.time()

# model.save('./_save/keras29_mcp/keras29_3_save_model.h5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score, accuracy_score

# y_test = y_test.to_numpy()
# y_predict = y_predict.to_numpy()

y_test = np.argmax(y_test, axis=1).reshape(-1,1)
y_predict = np.argmax(y_predict, axis=1).reshape(-1,1)

acc = accuracy_score(y_test, y_predict)

print('로스 : ', loss)
print('acc_score :', acc)
print(y_predict)
print(y_predict.shape)
print(y_predict[0])

# padding
# 로스 :  [0.2835237383842468, 0.9211999773979187]
# acc_score : 0.9212

# MaxPooling2D
# 로스 :  [0.08223845809698105, 0.9740999937057495]
# acc_score : 0.9741

# 함수로 변환 시
# 로스 :  [0.0803336575627327, 0.973800003528595]
# acc_score : 0.9738


# 데이터 증폭
# 로스 :  [0.082846499979496, 0.974399983882904]
# acc_score : 0.9744

# 이미지 반전 false
# 로스 :  [0.07905659824609756, 0.9765999913215637]
# acc_score : 0.9766
