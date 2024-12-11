#기존 캐글 데이터에서 
# # 1. train_csv의 y를 casul과 register로 잡는다.
# 그래서 훈련을 해서 test_csv의 casual과 register를 predict한다.

# 2. test_csv에 casual과 register 컬럼을 합쳐 #붙이는 것은 어떻게?

# 3. train_csv에 y를 count로 잡는다.

# 4. 전체 훈련

# 5. test_csv 예측해서 submission에 붙여! 

#파일 2번에 casual과 registered까지
#파일 3번에는?

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

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
print(x.shape)
y = train_csv[['casual', 'registered']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=10)
print(y.shape) #(10886, 2)

#2 모델구성
model = Sequential()
model.add(Dense(50, activation= 'relu', input_dim = 8))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(2, activation= 'linear'))

#3 컴파일 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=10)

#4 평가 예측
loss = model.evaluate(x_test, y_test)
y_submit = model.predict(test_csv)
print(y_submit.shape) # (6493,2)

print("test_csv타입 : ", type(test_csv))
print("y_submit타입 : ", type(y_submit))

test2_csv = test_csv
print(test2_csv.shape) # (6493, 8)

test2_csv[['casual', 'registered']] = y_submit #두 개 이상은 리스트 이기 때문에 리스트로 
print(test2_csv) # [6493 rows x 10 columns]

test2_csv.to_csv(path+ "test2.csv")


y_predict = model.predict([x_test])
r2 = r2_score(y_test, y_predict)

##### submission.csv 만들기###

#5. 파일 출력
# sampleSubmission['count'] = y_submit
# print(sampleSubmission)

# sampleSubmission.to_csv(path+'samplesubmission_0717_1413.csv') #to_csv는 이 데이터를 ~파일을 만들어서 거기에 넣어줄거임
print(loss)
