import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train/255.
x_test = x_test/255.

train_datagen = ImageDataGenerator(
    #rescale=1./255,
    horizontal_flip=True, # 수평 뒤집기
    vertical_flip=True, # 수직 뒤집기
    width_shift_range=0.2, # 평행이동 수평 이미지 전체를 10프로만큼 이동시켜준다.
    # height_shift_range=0.1, # 평행이동 수직
    rotation_range=15, #각도 만큼 이미지 회전
    # zoom_range=1.2, #축소 또는 확대
    # shear_range=0.7, # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환
    fill_mode='nearest', #이동했을 때 비는 공간을 가장 가까운 곳의 데이터로 채운다. 예를 들어 주변에 배경이 있다면 그 배경에 가까운 색으로 채워짐
    )  

augment_size = 40000 #늘리고 싶은 만큼 0-9까지의 라벨이 있음 각각 4000개씩 늘려주는 게 좋다. 60000개는 10개의 클래스에 담겨 있어서 6000개씩 나눠짐
#6000개 중에서 4000개를 뽑아서 그 중 1개씩만 늘려준다.

print(x_train.shape[0]) #60000
randidx = np.random.randint(x_train.shape[0], size=augment_size) #60000, size = 40000
print(randidx) #[59723 21087 24328 ... 47215 14156 23882]
print(np.min(randidx), np.max(randidx)) #2 59996

print(x_train[0].shape) #(28, 28)

x_augmented = x_train[randidx].copy() #.copy() 하면 변수에 새로 할당해서 원래의 x_train의 영향을 미치지 않는다.
y_augmented = y_train[randidx].copy() 

#4만개의 x와 y의 카피본 준비했음

print(x_augmented.shape, y_augmented.shape) #(40000, 28, 28) (40000,)

x_augmented = x_augmented.reshape(
    x_augmented.shape[0], #40000
    x_augmented.shape[1], #28
    x_augmented.shape[2], 1) #28, 1

print(x_augmented.shape)    #(40000, 28, 28, 1)


x_augmented = train_datagen.flow(
    x_augmented, y_augmented, 
    batch_size = augment_size,
    shuffle=False
).next()[0] #이렇게 사용해줄 수 있음

print(x_augmented.shape) # (40000, 28, 28, 1) *변환된 데이터 

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

print(x_train.shape, x_test.shape) #(60000, 28, 28, 1) (10000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augmented), axis=0)
print(x_train.shape) #(100000, 28, 28, 1)
y_train = np.concatenate((y_train, y_augmented), axis=0)
print(y_train.shape) #(100000,)

print(np.unique(y_train))

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

y_test = y_test.to_numpy()

### 맹그러봐###
# y에 남자 여자 데이터 불균형 할 시에 여자만 높여야함 np.unique 해보셈

model = Sequential()
model.add(Conv2D(64, (2,2), activation='relu', input_shape = (28,28,1),
                 strides=1,
                 padding='same')) 
model.add(MaxPooling2D())
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Conv2D(filters = 128, activation='relu', kernel_size= (2,2),
                 strides=1,
                 padding='same')) 
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), activation='relu',
                 strides=1,
                 padding='same')) 
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=1024, input_shape=(32,), activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(units=512, input_shape=(32,), activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(10, activation='softmax'))

#3 컴파일 훈련

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
start_time = time.time()

es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience=30,
    verbose=1,
    restore_best_weights=True
)

import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path1 = './_save/keras49/fashion/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path1, 'k30_', date, '_', filename])


mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose=1,
    save_best_only=True,
    filepath=filepath
    )


model.fit(x_train, y_train, epochs=500, batch_size=16, verbose=1, validation_split=0.25, callbacks=[es, mcp])
end_time = time.time()

# 예측, 평가
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

y_test = np.argmax(y_test, axis=1).reshape(-1, 1)
y_predict = np.argmax(y_predict, axis=1).reshape(-1, 1)

acc = accuracy_score(y_test, y_predict)

print('로스 : ', loss[0])
print('acc 스코어 :', round(acc, 3))
print('시간 : ', round(end_time-start_time, '초'))



# print(x_train.shape) #(60000, 28, 28)
# # print(x_train[0].shape) (28, 28)

# # aaa = x_train[0].reshape(28, 28, 1)

# # plt.imshow(aaa, cmap='gray')
# # plt.show()

# # aaa= np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1)
# # print(aaa.shape)

# xy_data = train_datagen.flow(
#     np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1),
#     np.zeros(augment_size),
#     batch_size=augment_size,
#     shuffle=False,
# ).next()

# print(xy_data)
# print(type(xy_data)) # <class 'tuple'> 

# # print(xy_data.shape)
# print(len(xy_data)) #2
# print(xy_data[0].shape) #(100, 28, 28, 1)
# print(xy_data[1].shape) #(100,)

# #증폭시킨 데이터를 변환시킬 것이다. 

# plt.figure(figsize=(7,7))
# for i in range(49):
#     plt.subplot(7, 7, i+1)
#     plt.imshow(xy_data[0][i], cmap='gray')
#     plt.axis('off')
# plt.show()

# 로스 :  0.24789275228977203
# acc 스코어 : 0.913