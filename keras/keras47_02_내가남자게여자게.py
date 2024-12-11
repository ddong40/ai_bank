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
np_path1 = 'C:/Users/ddong40/ai_2/_data/_save_npy/gender/'
x_test1 = np.load(np_path1 + 'keras43_01_x_test.npy')


np_path = 'C:/Users/ddong40/ai_2/_data/image/me/'
x_train = np.load(np_path + 'me.npy')

#평가 예측
model = load_model('_save/keras45/k30_0805_1458_0051-0.2962.hdf5')
# loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_train, batch_size=16)
acc = 1-y_predict

# y_test = np.argmax(y_test).reshape(-1, 1)
# y_predict = np.argmax(y_predict).reshape(-1, 1)

# acc = accuracy_score(y_test, y_predict)

# y_submit = model.predict(x_test1)

print(y_predict)

print('나는 {} 확률로 남자입니다.'.format(acc))