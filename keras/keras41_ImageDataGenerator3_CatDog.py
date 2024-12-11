#캣독 맹그러봐! 
# train_test_split해라 
# 1. 에서 시간 체크 

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time


train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True, # 수평 뒤집기
    vertical_flip=True, # 수직 뒤집기
    width_shift_range=0.1, # 평행이동 수평 이미지 전체를 10프로만큼 이동시켜준다.
    # height_shift_range=0.1, # 평행이동 수직
    rotation_range=0.1, #각도 만큼 이미지 회전
    # zoom_range=1.2, #축소 또는 확대
    # shear_range=0.7, # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환
    fill_mode='nearest', #이동했을 때 비는 공간을 가장 가까운 곳의 데이터로 채운다. 예를 들어 주변에 배경이 있다면 그 배경에 가까운 색으로 채워짐
    )  
start_time1 = time.time()
test_datagen = ImageDataGenerator(
    rescale=1./255)

path_train = './_data/image/cat_and_dog/Train/' #라벨이 분류된 상위 폴더까지 path를 잡는다. 이후 수치화하면 0과 1로 바뀐다.
path_test = './_data/image/cat_and_dog/Test/' 

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(80,80),
    batch_size=20000,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True) #imagedatagenertor방식으로 디렉토리부터 흘러와서 이미지를 수치화하여 xy_train에 담아라
# batch를 10으로 주면 16*(10, 200, 200, 1)
# train폴더에 ad와 normal이 각각 80개 

xy_test = test_datagen.flow_from_directory(
    path_train,
    target_size=(80,80),
    batch_size=20000,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False) #test

np_path = 'C:/Users/ddong40/ai_2/_data/_save_npy/cat_dog/'
np.save(np_path + 'keras43_01_x_train.npy', arr=xy_train[0][0]) # 이 경로에 x_train 데이터를 저장
np.save(np_path + 'keras43_01_y_train.npy', arr=xy_train[0][1]) # 이 경로에 y_train 데이터 저장
np.save(np_path + 'keras43_01_x_test.npy', arr=xy_test[0][0]) # 이 경로에 x_test 데이터를 저장
np.save(np_path + 'keras43_01_y_test.npy', arr=xy_test[0][1]) # 이 경로에 y_test 데이터를 저장
# train test split 이전에 저장해줄 것 
'''
x_train = xy_train[0][0] 
y_train = xy_train[0][1] 
x_test1 = xy_test[0][0]
y_test1 = xy_test[0][1]

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, shuffle=True, random_state=3)
end_time1 = time.time()
print(round(end_time1 - start_time1, 3), '초')

print(x_train.shape) #(13997, 100, 100, 3)
print(y_train.shape) #(13997,)
print(y_train[0]) #1.0

#모델 구성

model = Sequential()
model.add(Conv2D(32, 2, input_shape=(100, 100, 3), padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Conv2D(32, 2, activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, 2, activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, 2, activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
start_time = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 20,
    restore_best_weights=True,
    verbose=1
)

import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path1 = './_save/keras41/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path1, 'k30_', date, '_', filename])


mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only=True,
    verbose=1,
    filepath=filepath
)

model.fit(x_train, y_train, epochs= 300, batch_size=100, verbose=1, validation_split=0.25, callbacks=[es, mcp])
end_time = time.time()

#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

y_test = np.argmax(y_test).reshape(-1, 1)
y_predict = np.argmax(y_predict).reshape(-1, 1)

acc = accuracy_score(y_test, y_predict)

print('로스 : ', loss[0])
print('정확도 : ', loss[1])
print('시간 :', round(end_time - start_time, 3), '초')

# 로스 :  0.6931867003440857
# 정확도 :  0.4963333308696747
# 시간 : 208.842 초

# batch 100, 
# 로스 :  0.5225458145141602
# 정확도 :  0.7413333058357239
# 시간 : 101.995 초

# 로스 :  0.5298580527305603
# 정확도 :  0.734000027179718
# 시간 : 117.387 초
'''