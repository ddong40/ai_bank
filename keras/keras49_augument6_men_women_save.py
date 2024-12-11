# https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True, # 수평 뒤집기
    vertical_flip=True, # 수직 뒤집기
    width_shift_range=0.1, # 평행이동 수평 이미지 전체를 10프로만큼 이동시켜준다.
    height_shift_range=0.1, # 평행이동 수직
    rotation_range=1, #각도 만큼 이미지 회전 / 1
    zoom_range=0.2, #축소 또는 확대 0.2로 하면 / 0.8 - 1.2 사이로 무작위
    shear_range=0.7, # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환
    fill_mode='nearest' #이동했을 때 비는 공간을 가장 가까운 곳의 데이터로 채운다. 예를 들어 주변에 배경이 있다면 그 배경에 가까운 색으로 채워짐
    )  
 
# 이 데이터들은 훈련에서 사용될 것 들이다.

test_datagen = ImageDataGenerator(
    rescale=1./255)

path_train = 'C:/Users/ddong40/ai_2/_data/kaggle/biggest gender/faces/' #라벨이 분류된 상위 폴더까지 path를 잡는다. 이후 수치화하면 0과 1로 바뀐다.
path_test = 'C:/Users/ddong40/ai_2/_data/kaggle/biggest gender/faces/' 

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(100,100),
    batch_size=27167,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False) #imagedatagenertor방식으로 디렉토리부터 흘러와서 이미지를 수치화하여 xy_train에 담아라
# batch를 10으로 주면 16*(10, 200, 200, 1)
# train폴더에 ad와 normal이 각각 80개 


xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(100,100),
    batch_size=27167,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False) #test

np_path = 'C:/Users/ddong40/ai_2/_data/_save_npy/gender/gender2/'
np.save(np_path + 'keras43_01_x_train_man.npy', arr=xy_train[0][0][:17678]) # 이 경로에 x_train 데이터를 저장
np.save(np_path + 'keras43_01_y_train_man.npy', arr=xy_train[0][1][:17678])
np.save(np_path + 'keras43_01_x_train_woman.npy', arr=xy_train[0][0][17678:]) # 이 경로에 x_train 데이터를 저장
np.save(np_path + 'keras43_01_y_train_woman.npy', arr=xy_train[0][1][17678:])
np.save(np_path + 'keras43_01_x_test.npy', arr=xy_test[0][0]) # 이 경로에 x_test 데이터를 저장
np.save(np_path + 'keras43_01_y_test.npy', arr=xy_test[0][1]) # 이 경로에 y_test 데이터를 저장