# Global average pooling 
# 전부의 평균 값을 뽑아서 풀링함

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10

# (x_train, y_train), (x_test, y_test) = cifar10.load_data() 

# x_train = x_train/255.
# x_test = x_test/255.

# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(sparse=False)
# # y_train = y_train.reshape(-1, 1)
# # y_test = y_test.reshape(-1, 1)
# y_train = ohe.fit_transform(y_train)
# y_test = ohe.transform(y_test)


vgg16 = VGG16(#weights='imagenet',
              include_top=False,          
              input_shape=(224, 224, 3))    

# vgg16.trainable = True

model = Sequential()
model.add(vgg16)
model.add(GlobalAveragePooling2D())
# model.add(Flatten())

model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.summary()
