import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

np_path = 'C:/Users/ddong40/ai_2/_data/_save_npy/gender/gender2/'
x_train = np.load(np_path + 'keras43_01_x_train_man.npy' )
y_train = np.load(np_path + 'keras43_01_y_train_man.npy')
x_train2 = np.load(np_path + 'keras43_01_x_train_woman.npy')
y_train2 = np.load(np_path + 'keras43_01_y_train_woman.npy')
x_test1 = np.load(np_path + 'keras43_01_x_test.npy')
y_test1 = np.load(np_path + 'keras43_01_y_test.npy')


print(x_train.shape)
print(y_train.shape)
print(x_train2.shape)
print(y_train2.shape)
# print(x_test1.shape)
# print(y_test1.shape)

datagen = ImageDataGenerator(
    # rescale=1./255,
    horizontal_flip=False, # 수평 뒤집기
    vertical_flip=False, # 수직 뒤집기
    # width_shift_range=0.1, # 평행이동 수평 이미지 전체를 10프로만큼 이동시켜준다.
    # height_shift_range=0.1, # 평행이동 수직
    # rotation_range=1, #각도 만큼 이미지 회전 / 1
    # zoom_range=0.01, #축소 또는 확대 0.2로 하면 / 0.8 - 1.2 사이로 무작위
    # shear_range=0.7, # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환
    fill_mode='nearest' #이동했을 때 비는 공간을 가장 가까운 곳의 데이터로 채운다. 예를 들어 주변에 배경이 있다면 그 배경에 가까운 색으로 채워짐
    )  


augment_size = 8189

randidx = np.random.randint(x_train2.shape[0], size = augment_size)

x_augmented = x_train2[randidx].copy()
y_augmented = y_train2[randidx].copy()


x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2],3)

print(x_augmented.shape)
print(y_augmented.shape)


x_augmented = datagen.flow(
    x_augmented, y_augmented,
    batch_size= augment_size,
    shuffle = False,
    save_to_dir='C:/Users/ddong40/ai_2/_data/_save_img/06_men_women/'
).next()[0]

# x_train = x_train.reshape(50000, 32,32,3)
# x_test = x_test.reshape(10000, 32,32,3)

print(x_augmented.shape)
print(y_augmented.shape)
'''
x_train2 = np.concatenate((x_train2, x_augmented), axis=0)
y_train2 = np.concatenate((y_train2, y_augmented), axis=0)

print('치킨')
print(x_train2.shape)
print(y_train2.shape)

x_train = np.concatenate((x_train, x_train2), axis=0)
y_train = np.concatenate((y_train, y_train2), axis=0)

print(x_train.shape)
print(y_train.shape)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=100, shuffle=True)


# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)

# 모델 구성
model = Sequential()
model.add(Conv2D(32, 2, activation='relu', input_shape = (100, 100, 3), padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(64, 2, activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(128, 2, activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(64, 2, activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))

# 컴파일 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])

start_time = time.time()

import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path1 = 'C:/Users/ddong40/ai_2/_save/keras49/men_women/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path1, 'k30_', date, '_', filename])


es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    verbose=1,
    patience=30,
    restore_best_weights=True   
)

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'min',
    verbose=1,
    save_best_only=True,
    filepath = filepath
)


model.fit(x_train, y_train, epochs=100, batch_size=16, verbose=1, validation_split=0.25, callbacks= [es, mcp] )              

end_time = time.time()

# 평가 예측
loss = model.evaluate(x_test, y_test, batch_size=16)
y_predict = model.predict(x_test, batch_size=16)

print('로스 : ', loss[0])
print('acc : ', loss[1])
print('시간 : ', round(end_time - start_time, 2), '초')

# 로스 :  0.29976391792297363
# acc :  0.867500901222229
# 시간 :  944.158 초

# 로스 :  0.21354340016841888
# acc :  0.9244909286499023
# 시간 :  1056.64 초
'''