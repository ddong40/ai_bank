# https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time 

#1. 데이터
path = 'C:/Users/ddong40/ai/_data/bike-sharing-demand/' #절대경로(경로가 풀로 다 들어간 경우)
# path = 'C:/Users/ddong40/ai/_data/bike-sharing-demand' #위와 다 같음
# path = 'C://Users//ddong40//ai//_data//bike-sharing-demand' #위와 다 같음

train_csv = pd.read_csv(path + "train.csv", index_col = 0)
test_csv = pd.read_csv(path + "test.csv", index_col = 0)
sampleSubmission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

print(train_csv.shape) # (10886, 11)
print(test_csv.shape) # (6493, 8)
print(sampleSubmission.shape) #(6493, 1)

print(train_csv.columns)
# ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    #    'humidity', 'windspeed', 'casual', 'registered', 'count'],
    #   dtype='object'
    
print(train_csv.info())
print(test_csv.info())

print(train_csv.describe())

######### 결측치 확인 ###########

print(train_csv.isna().sum())
print(train_csv.isnull().sum())
print(test_csv.isna().sum())
print(test_csv.isnull().sum())


###### x와 y분리
x = train_csv.drop(['casual','registered','count'], axis=1) #이 리스트의 컬럼들을 axis 1에 넣어 드랍해주세요 라는 뜻
print(x)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=10)

print(x_train.shape)

#2 모델구성
model = Sequential()
model.add(Dense(100, activation= 'relu', input_dim = 8))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(1, activation= 'linear'))




#3 컴파일 훈련
model.compile(loss = 'mse', optimizer='adam')
start = time.time()

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience = 10,
    restore_best_weights = True 
)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=2, validation_split=0.2,
                 callbacks=[es])
end = time.time()

#4 평가 예측
loss = model.evaluate(x_test, y_test)
y_submit = model.predict(test_csv)
y_predict = model.predict([x_test])
r2 = r2_score(y_test, y_predict)

#5. 파일 출력
sampleSubmission['count'] = y_submit
print(sampleSubmission)

sampleSubmission.to_csv(path+'samplesubmission_0717_1413.csv') #to_csv는 이 데이터를 ~파일을 만들어서 거기에 넣어줄거임
print(loss)

print(hist)
print('======================= hist.history==================')
print(hist.history)
print('================loss=================')
print(hist.history['loss'])
print('=================val_loss==============')
print(hist.history['val_loss'])
print('====================================================')


import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.legend(loc='upper right') #라벨 값이 무엇인지 명시해주는 것이 레전드
plt.title('캐글 Loss') #그래프의 제목 
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()

