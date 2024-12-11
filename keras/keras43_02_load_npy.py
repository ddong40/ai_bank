#데이터를 넘파이로 저장하기

# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Input, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

import os
import natsort

# file_path = "C:/Users/ddong40/ai_2/_data/kaggle/dogs-vs-cats-redux-kernels-edition/test/test/"
# file_names = natsort.natsorted(os.listdir(file_path))

# print(np.unique(file_names))
# i = 1
# for name in file_names:
#     src = os.path.join(file_path,name)
#     dst = str(i).zfill(5)+ '.jpg'
#     dst = os.path.join(file_path, dst)
#     os.rename(src, dst)
#     i += 1

start = time.time()



np_path = 'C:/Users/ddong40/ai_2/_data/_save_npy/'
# np.save(np_path + 'keras43_01_x_train.npy', arr=xy_train[0][0]) # 이 경로에 x_train 데이터를 저장
# np.save(np_path + 'keras43_01_x_train.npy', arr=xy_train[0][1]) # 이 경로에 y_train 데이터 저장
# np.save(np_path + 'keras43_01_x_test.npy', arr=xy_test[0][0]) # 이 경로에 x_test 데이터를 저장
# np.save(np_path + 'keras43_01_x_test.npy', arr=xy_test[0][1]) # 이 경로에 y_test 데이터를 저장
# # train test split 이전에 저장해줄 것 

x_train = np.load(np_path + 'keras43_01_x_train.npy' )
y_train = np.load(np_path + 'keras43_01_y_train.npy')
x_test = np.load(np_path + 'keras43_01_x_test.npy')
y_test = np.load(np_path + 'keras43_01_y_test.npy')

# print(x_train)
print(y_train)
print(y_train.shape) #[1. 1. 0. ... 0. 1. 1.]
# (25000,)
print(x_test)
print(x_test.shape)




# x = xy_train[0][0]
# y = xy_train[0][1]

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1024)

# x_test1 = xy_test[0][0]


# print(x_train.shape) #(25000, 100, 100, 3)



end = time.time()
print('시간 :', round(end - start, 3), '초')

# # 리스트 안에 수치들은 넘파이로 변환해서 사용한다. 
# # 모든 수치들은 넘파이다. 

# print(y_train.shape) #(25000, )
# print(y_train)

