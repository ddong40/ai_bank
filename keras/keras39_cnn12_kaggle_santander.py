#https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
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

print(x_train.shape)
print(x_test.shape)

x_train = x_train.reshape(140000, 20, 10, 1)
x_test = x_test.reshape(60000, 20, 10, 1)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# #2 모델구성

# # model = Sequential()
# # model.add(Dense(128, activation = 'relu', input_dim = 200))
# # model.add(Dropout(0.2))
# # model.add(Dense(128, activation = 'relu'))
# # model.add(Dropout(0.2))
# # model.add(Dense(128, activation = 'relu'))
# # model.add(Dropout(0.2))
# # model.add(Dense(128, activation = 'relu'))
# # model.add(Dropout(0.2))
# # model.add(Dense(64, activation = 'relu'))
# # model.add(Dropout(0.2))
# # model.add(Dense(64, activation = 'relu'))
# # model.add(Dropout(0.2))
# # model.add(Dense(64, activation = 'relu'))
# # model.add(Dropout(0.2))
# # model.add(Dense(64, activation = 'relu'))
# # model.add(Dropout(0.2))
# # model.add(Dense(32, activation = 'relu'))
# # model.add(Dropout(0.2))
# # model.add(Dense(32, activation = 'relu'))
# # model.add(Dropout(0.2))
# # model.add(Dense(32, activation = 'relu'))
# # model.add(Dropout(0.2))
# # model.add(Dense(32, activation = 'relu'))
# # model.add(Dropout(0.2))
# # model.add(Dense(1, activation = 'sigmoid'))

model = Sequential()
model.add(Conv2D(32, 2, activation= 'relu', input_shape = (20, 10, 1)))
model.add(MaxPooling2D())
model.add(Conv2D(32, 2, activation= 'relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32, 2, activation= 'relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(32, activation='relu', input_shape = (32,)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))


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

path1 = './_save/keras40/12_kaggle_santander_customer/'
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


# y_submit = model.predict(test_csv)
# y_submit = np.round(y_submit)

# sampleSubmission['target'] = y_submit

# sampleSubmission.to_csv(path+'samplesubmission_0724_1520.csv') 

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

# dnn을 cnn으로 변환
# 로스 :  0.25406137108802795
# 정확도 :  0.906
# 시간 2712.52 초