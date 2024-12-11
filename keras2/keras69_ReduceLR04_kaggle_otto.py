#https://www.kaggle.com/competitions/otto-group-product-classification-challenge/data

#0.89점 이상 뽑아내기 ㅎㅎ


import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import random as rn
rn.seed(337)
tf.random.set_seed(337)
np.random.seed(337)


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

print(x_test.shape)
print(y_test.shape)

#2 모델구성

model = Sequential()
model.add(Dense(512, activation= 'relu', input_dim = 93))
model.add(Dropout(0.3))
model.add(Dense(512, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(9, activation= 'softmax'))

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
lr = np.array([0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001])



#3 컴파일 훈련

for i in lr:
    

    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate=i), metrics = ['accuracy'])

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
        verbose = 0,
        save_best_only=True,
        filepath = filepath
    )
    rlr = ReduceLROnPlateau(
        monitor = 'val_loss',
        mode = 'auto',
        verbose = 0,
        patience= 15,
        factor= 0.9
    )

    model.fit(x_train, y_train, epochs = 1, batch_size= 128, verbose = 0, validation_split=0.25, callbacks = [es, mcp, rlr])

    end_time = time.time()

    #4 평가 예측
    loss = model.evaluate(x_test, y_test, verbose=0)
    # print('로스 값 :',loss[0])
    # print('정확도 : ',round(loss[1],3))
    # print('시간 : ', round(end_time - start_time, 2), '초')
    y_pred = model.predict(x_test)
    y_submit = model.predict(test_csv)
    y_submit = np.round(y_submit)

    # sampleSubmission[['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']] = y_submit


    # sampleSubmission.to_csv(path + 'sampleSubmission_0724_1849.csv')
    print('{0} : '.format(i))
    print("로스값 : ", loss[0])
    print("정확도 : ", loss[1])
    print('----------------------------')
    

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