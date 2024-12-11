import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

np_path = 'C:/Users/ddong40/ai_2/_data/_save_npy/horse/'
# np.save(np_path + 'keras43_01_x_train.npy', arr=xy_train[0][0]) # 이 경로에 x_train 데이터를 저장
# np.save(np_path + 'keras43_01_y_train.npy', arr=xy_train[0][1]) # 이 경로에 y_train 데이터 저장
# np.save(np_path + 'keras43_01_x_test.npy', arr=xy_test[0][0]) # 이 경로에 x_test 데이터를 저장
# np.save(np_path + 'keras43_01_y_test.npy', arr=xy_test[0][1]) # 이 경로에 y_test 데이터를 저장
x_train = np.load(np_path + 'keras43_01_x_train.npy' )
y_train = np.load(np_path + 'keras43_01_y_train.npy')

datagen = ImageDataGenerator(horizontal_flip=False,
                             vertical_flip=True,
                             fill_mode='nearest')

print(x_train.shape)
print(y_train.shape)

augument_size = 8973

randidx = np.random.randint(x_train.shape[0], size = augument_size)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

print(x_augmented.shape)

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2],3)

print(x_augmented.shape)

x_augmented = datagen.flow(
    x_augmented, y_augmented,
    batch_size=augument_size,
    shuffle=False 
).next()[0]

print(x_augmented.shape)
print(y_augmented.shape)

x_train = np.concatenate((x_train, x_augmented), axis=0)
y_train = np.concatenate((y_train, y_augmented), axis=0)

print(x_train.shape)
print(y_train.shape)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=100)

model = load_model('_save/keras41/horse/k30_0805_1215_0064-0.0026.hdf5')
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

y_test = np.argmax(y_test).reshape(-1, 1)
y_predict = np.argmax(y_predict).reshape(-1, 1)

acc = accuracy_score(y_test, y_predict)

print('로스 : ', loss[0])
print('정확도 : ', loss[1])

# 증폭 전
# 로스 :  0.02543286606669426
# 정확도 :  0.9935275316238403


# 증폭 후 
# 로스 :  1.2745141983032227
# 정확도 :  0.8493333458900452

