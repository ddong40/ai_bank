#https://www.kaggle.com/competitions/otto-group-product-classification-challenge/data

#0.89점 이상 뽑아내기 ㅎㅎ


import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

#1 데이터
path = 'C:/Users/ddong40/ai_2/_data/kaggle/otto-group-product-classification-challenge/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
sampleSubmission = pd.read_csv(path + 'sampleSubmission.csv', index_col = 0)

encoder = LabelEncoder()

train_csv['target'] = encoder.fit_transform(train_csv['target'])

x = train_csv.drop('target', axis=1)
y = train_csv['target']

print(x.shape)
print(y.shape)

y = pd.get_dummies(y)

print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, shuffle= True, random_state= 10)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

print(x_train.shape) #(43314, 93)
print(x_test.shape) #(18564, 93)

x_train = x_train.reshape(43314, 93, 1, 1)
x_test = x_test.reshape(18564, 93, 1, 1)

#2 모델구성

# model = Sequential()
# model.add(Dense(512, activation= 'relu', input_dim = 93))
# model.add(Dropout(0.3))
# model.add(Dense(512, activation= 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(512, activation= 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(512, activation= 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(512, activation= 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(256, activation= 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(256, activation= 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(256, activation= 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(256, activation= 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation= 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation= 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation= 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation= 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(64, activation= 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(64, activation= 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(64, activation= 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(64, activation= 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(32, activation= 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(32, activation= 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(32, activation= 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(32, activation= 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(16, activation= 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(16, activation= 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(16, activation= 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(9, activation= 'softmax'))

# input1 = Input(shape=(93))
# dense1 = Dense(512, activation = 'relu')(input1)
# drop1 = Dropout(0.3)(dense1)
# dense2 = Dense(512, activation = 'relu')(drop1)
# drop2 = Dropout(0.3)(dense2)
# dense3 = Dense(512, activation = 'relu')(drop2)
# drop3 = Dropout(0.3)(dense3)
# dense4 = Dense(512, activation = 'relu')(drop3)
# drop4 = Dropout(0.3)(dense4)
# dense5 = Dense(512, activation = 'relu')(drop4)
# drop5 = Dropout(0.3)(dense5)
# dense6 = Dense(256, activation = 'relu')(drop5)
# drop6 = Dropout(0.3)(dense6)
# dense7 = Dense(256, activation = 'relu')(drop6)
# drop7 = Dropout(0.3)(dense7)
# dense8 = Dense(256, activation = 'relu')(drop7)
# drop8 = Dropout(0.3)(dense7)
# dense9 = Dense(256, activation = 'relu')(drop8)
# drop9 = Dropout(0.3)(dense9)
# dense10 = Dense(128, activation = 'relu')(drop9)
# drop10 = Dropout(0.3)(dense10)
# dense11 = Dense(128, activation = 'relu')(drop10)
# drop11 = Dropout(0.3)(dense11)
# dense12 = Dense(128, activation = 'relu')(drop11)
# drop12 = Dropout(0.3)(dense12)
# dense13 = Dense(128, activation = 'relu')(drop12)
# drop13 = Dropout(0.3)(dense13)
# dense14 = Dense(64, activation = 'relu')(drop13)
# drop14 = Dropout(0.3)(dense14)
# dense15 = Dense(64, activation = 'relu')(drop14)
# drop15 = Dropout(0.3)(dense15)
# dense16 = Dense(64, activation = 'relu')(drop15)
# drop16 = Dropout(0.3)(dense16)
# dense17 = Dense(64, activation = 'relu')(drop16)
# drop17 = Dropout(0.3)(dense17)
# dense18 = Dense(32, activation = 'relu')(drop17)
# drop18 = Dropout(0.3)(dense18)
# dense19 = Dense(32, activation = 'relu')(drop18)
# drop19 = Dropout(0.3)(dense19)
# dense20 = Dense(32, activation = 'relu')(drop19)
# drop20 = Dropout(0.3)(dense20)
# dense21 = Dense(32, activation = 'relu')(drop20)
# drop21 = Dropout(0.3)(dense21)
# dense22 = Dense(16, activation = 'relu')(drop21)
# drop22 = Dropout(0.3)(dense22)
# dense23 = Dense(16, activation = 'relu')(drop22)
# drop23 = Dropout(0.3)(dense23)
# dense24 = Dense(16, activation = 'relu')(drop23)
# drop24 = Dropout(0.3)(dense24)
# output1 = Dense(9, activation = 'softmax')(drop24)
# model = Model(inputs = input1, outputs = output1)

model = Sequential()
model.add(Conv2D(32, 2, activation= 'relu', input_shape = (93, 1, 1), padding='same'))
model.add(Conv2D(32, 2, activation= 'relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, 2, activation= 'relu', padding='same'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(32, activation='relu', input_shape = (32,)))
model.add(Dropout(0.2))
model.add(Dense(9, activation='softmax'))


#3 컴파일 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

start_time = time.time()

es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience= 30,
    restore_best_weights= True
)
import datetime 
date = datetime.datetime.now() #데이트라는 변수에 현재 시간을 반환한다.
print(date) #2024-07-26 16:49:51.174797
print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M") #시간을 문자열로 바꿔줌
print(date) #0726_1654
print(type(date))

path1 = './_save/keras32/13_kaggle_otto/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # '1000-0.7777.hdf5'  #fit에서 반환되는 값을 빼오는 것이다. 
filepath = "".join([path1, 'k30_', date, '_', filename])

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose = 1,
    save_best_only=True,
    filepath = filepath
)

model.fit(x_train, y_train, epochs = 1000, batch_size= 50, verbose = 1, validation_split=0.25, callbacks = [es, mcp])

end_time = time.time()

#4 평가 예측
loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
print('로스 값 :',loss[0])
print('정확도 : ',round(loss[1],3))
print('시간 : ', round(end_time - start_time, 2), '초')

# y_submit = model.predict(test_csv)
# y_submit = np.round(y_submit)

# sampleSubmission[['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']] = y_submit


# sampleSubmission.to_csv(path + 'sampleSubmission_0724_1849.csv')

# 로스 값 : 0.6220168471336365
# 정확도 :  0.773

# 로스 값 : 0.600717306137085
# 정확도 :  0.787

# 로스 값 : 0.6098398566246033
# 정확도 :  0.792

# minmax
# 로스 값 : 0.6758685111999512
# 정확도 :  0.763

# StandardScaler


#세이브 값
# 로스 값 : 0.7251054048538208
# 정확도 :  0.758

# drop out
# 로스 값 : 1.293474555015564
# 정확도 :  0.516

#cpu
# 로스 값 : 1.3126633167266846
# 정확도 :  0.538
# 시간 :  290.5 초



#gpu
# 로스 값 : 1.2036213874816895
# 정확도 :  0.564
# 시간 :  526.87 초

# dnn - cnn으로 변환
# 로스 값 : 0.5477306842803955
# 정확도 :  0.799
# 시간 :  147.92 초