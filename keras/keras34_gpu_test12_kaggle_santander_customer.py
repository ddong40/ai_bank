#https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time


path = 'C:/Users/ddong40/ai_2/_data/kaggle/santander-customer-transaction-prediction/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sampleSubmission = pd.read_csv(path + 'sample_submission.csv', index_col=0)

x = train_csv.drop('target', axis=1)
y = train_csv['target']

print(x.shape)
print(y.shape)

# y = pd.get_dummies(y)

print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=1542, stratify=y)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

#2 모델구성

# model = Sequential()
# model.add(Dense(128, activation = 'relu', input_dim = 200))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation = 'sigmoid'))

input1 = Input(shape=(200))
dense1 = Dense(128, activation = 'relu')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(128, activation = 'relu')(drop1)
drop2 = Dropout(0.2)(dense2)
dense3 = Dense(128, activation = 'relu')(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(128, activation = 'relu')(drop3)
drop4 = Dropout(0.2)(dense4)
dense5 = Dense(64, activation = 'relu')(drop4)
drop5 = Dropout(0.2)(dense5)
dense6 = Dense(64, activation = 'relu')(drop5)
drop6 = Dropout(0.2)(dense6)
dense7 = Dense(64, activation = 'relu')(drop6)
drop7 = Dropout(0.2)(dense7)
dense8 = Dense(64, activation = 'relu')(drop7)
drop8 = Dropout(0.2)(dense8)
dense9 = Dense(32, activation = 'relu')(drop8)
drop9 = Dropout(0.2)(dense9)
dense10 = Dense(32, activation = 'relu')(drop9)
drop10 = Dropout(0.2)(dense10)
dense11 = Dense(32, activation = 'relu')(drop10)
drop11 = Dropout(0.2)(dense11)
dense12 = Dense(32, activation = 'relu')(drop11)
drop12 = Dropout(0.2)(dense12)
output1 = Dense(1, activation = 'sigmoid')(drop12)
model = Model(inputs = input1, outputs = output1)

# 3 컴파일 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

start_time = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 30,
    restore_best_weights=True
)
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path1 = './_save/keras32/12_kaggle_santander_customer/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([path1, 'k30_', date, '_', filename])

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose =1,
    save_best_only=True,
    filepath = filepath
)

model.fit(x_train, y_train, epochs=1000, batch_size=10, verbose=1, validation_split=0.25, callbacks=[es, mcp])

end_time = time.time()

#4 평가 예측
loss = model.evaluate(x_test, y_test)
print("로스 : ", loss[0])
print("정확도 : ", round(loss[1], 3))
print("시간", round(end_time - start_time, 2), '초')

y_pred = model.predict(x_test)


y_submit = model.predict(test_csv)
y_submit = np.round(y_submit)

sampleSubmission['target'] = y_submit

sampleSubmission.to_csv(path+'samplesubmission_0724_1520.csv') 

# 로스 :  0.24495652318000793
# 정확도 :  0.911

# minmaxscaler
# 로스값 :  0.13892140984535217
# 정확도 :  0.953

# standard scalering
# 로스 값 : 0.7609817981719971
# 정확도 :  0.714 

# MaxAbsScaler
# 로스 :  0.24586938321590424
# 정확도 :  0.911

# 세이브 값
# 로스 :  0.24112220108509064
# 정확도 :  0.911

# drop out
# 로스 :  0.3261842727661133
# 정확도 :  0.9

# cpu
# 로스 :  0.2630799412727356
# 정확도 :  0.9
# 시간 364.51 초

# gpu
# 로스 :  0.2645145654678345
# 정확도 :  0.9
# 시간 5441.09 초