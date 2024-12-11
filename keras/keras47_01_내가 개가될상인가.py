# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Input, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

import os
import natsort

np_path = 'C:/Users/ddong40/ai_2/_data/image/me/'
x_train = np.load(np_path + 'me.npy')



#평가 예측
model = load_model('_save/keras42/k30_0804_2307_0012-0.6465.hdf5')
# loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_train)
acc = 1-y_predict

# y_test = np.argmax(y_test).reshape(-1, 1)
y_predict = np.argmax(y_predict).reshape(-1, 1)

# acc = accuracy_score(y_test, y_predict)

# y_submit = model.predict(x_test1)

print(y_predict)
print('나는 {} 확률로 개입니다.'.format(acc))

# print('로스 : ', loss[0])
# print('정확도 : ', loss[1])
# print('시간 :', round(end_time - start_time, 3), '초')


# #5. 파일 출력
# sampleSubmission['label'] = y_submit
# # print(sampleSubmission)

# sampleSubmission.to_csv(path+'samplesubmission_0804_1148.csv') #to_csv는 이 데이터를 ~파일을 만들어서 거기에 넣어줄거임


# print(xy_train.class_indices)

# [[0]]
# 나는 [[0.99999994]] 확률로 개입니다.