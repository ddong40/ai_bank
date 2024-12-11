from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import time
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import tensorflow as tf
import random as rn 
rn.seed(337)
tf.random.set_seed(337)
np.random.seed(337)

datasets = fetch_covtype()

x = datasets.data
y = datasets.target

print(datasets.DESCR)

# from tensorflow.keras.utils import to_categorical #케라스 (581012, 8)
# y = to_categorical(y) 

# y = pd.get_dummies(y) #(581012, 7)


y = y.reshape(-1,1) #(581012, 7)
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(y)

print(x.shape)
print(y.shape) #(581012, 7)

# print(pd.value_counts(y,))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, 
                                                    shuffle=True, random_state=500,
                                                    stratify=y)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


print(x_train.shape, y_train.shape) #406708, 54
print(x_test.shape, y_test.shape) #174304, 54

x_train = x_train.reshape(406708, 9, 6, 1)
x_test = x_test.reshape(174304, 9, 6, 1)

# print(pd.value_counts(y_train,))

#2. 모델 구성

# model = Sequential()
# model.add(Dense(128, activation= 'relu', input_dim = 54))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(7, activation='softmax'))

# input1 = Input(shape=(54))
# dense1 = Dense(128, activation = 'relu')(input1)
# drop1 = Dropout(0.3)(dense1)
# dense2 = Dense(128, activation = 'relu')(drop1)
# drop2 = Dropout(0.3)(dense2)
# dense3 = Dense(128, activation = 'relu')(drop2)
# drop3 = Dropout(0.3)(dense3)
# dense4 = Dense(128, activation = 'relu')(drop3)
# drop4 = Dropout(0.3)(dense4)
# dense5 = Dense(64, activation = 'relu')(drop4)
# drop5 = Dropout(0.3)(dense5)
# dense6 = Dense(64, activation = 'relu')(drop5)
# drop6 = Dropout(0.3)(dense6)
# dense7 = Dense(64, activation = 'relu')(drop6)
# drop7 = Dropout(0.3)(dense7)
# dense8 = Dense(32, activation = 'relu')(drop7)
# drop8 = Dropout(0.3)(dense7)
# dense9 = Dense(32, activation = 'relu')(drop8)
# drop9 = Dropout(0.3)(dense9)
# dense10 = Dense(32, activation = 'relu')(drop9)
# drop10 = Dropout(0.3)(dense10)
# output1 = Dense(7, activation = 'softmax')(drop10)
# model = Model(inputs = input1, outputs = output1)

model = Sequential()
model.add(Conv2D(32, 2, activation='relu', input_shape=(9,6,1), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32 ,2, activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32 ,2, activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32 ,2, activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(32, activation='relu', input_shape=(32,)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation = 'softmax'))

model.summary()


#3 컴파일 훈련

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

lr = np.array([0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001])

for i in lr:
     

    model.compile(loss = 'categorical_crossentropy', optimizer=Adam(learning_rate=i), metrics=['accuracy'])

    start_time = time.time()

    import datetime
    date = datetime.datetime.now()
    date = date.strftime('%m%d_%H%M')

    es = EarlyStopping(
        monitor = 'val_loss',
        mode = 'min',
        patience = 30,
        restore_best_weights=True
        )
    
    rlr = ReduceLROnPlateau(
    monitor = 'val_loss',
    mode = 'auto',
    patience= 15,
    verbose=1,
    factor = 0.9)   #factor는 learning rate * factor값

    path1 = './_save/keras39/10_fetch_cotype/'
    filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
    filepath = "".join([path1, 'k30_', date])


    mcp = ModelCheckpoint(
        monitor = 'val_loss',
        mode = 'auto',
        verbose = 0,
        save_best_only=True,
        filepath = filepath
    )


    model.fit(x_train, y_train, epochs=1, batch_size=300, verbose=0 , validation_split=0.25, callbacks=[es, mcp, rlr])
    end_time = time.time()

    #평가 예측
    loss = model.evaluate(x_test, y_test, verbose=0)
    y_pred = model.predict(x_test)
    print('{0} : '.format(i))
    print("로스값 : ", loss[0])
    print("정확도 : ", loss[1])
    print('----------------------------')
   





# 로스값 :  0.22436301410198212
# 정확도 :  0.911

# 로스값 :  0.21188057959079742
# 정확도 :  0.918
# batch_size=200
# patience = 30

# 로스값 :  0.17798465490341187
# 정확도 :  0.934
# batch_size=300
# patience = 50

# minmaxscaler
# 로스값 :  0.14274589717388153
# 정확도 :  0.95

# standardscaler


# maxabscaler


# RobustScaler
# 로스값 :  0.13968294858932495
# 정확도 :  0.954

# 세이브점수
# 로스 :  1.2455443143844604
# 정확도 :  0.961

# dropout 
# 로스값 :  0.5264937877655029
# 정확도 :  0.777

# cpu
# 로스값 :  0.3416094183921814
# 정확도 :  0.871

# gpu
# 로스값 :  0.3438628017902374
# 정확도 :  0.866

# dnn을 cnn으로 변환
# 로스값 :  0.37021759152412415
# 정확도 :  0.85
