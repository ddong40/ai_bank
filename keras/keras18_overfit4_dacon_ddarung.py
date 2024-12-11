import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
path = 'C:/Users/ddong40/ai_2/_data/따릉이/'       # 경로지정 #상대경로 

train_csv = pd.read_csv(path + "train.csv", index_col=0)   # . = 루트 = AI5 폴더,  index_col=0 첫번째 열은 데이터가 아니라고 해줘
print(train_csv)     # [1459 rows x 10 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0) #predict할 x데이터
print(test_csv)     # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + "submission.csv", index_col=0) #predict 할 y데이터
print(submission_csv)       #[715 rows x 1 columns],    NaN = 빠진 데이터
# 항상 오타, 경로 , shape 조심 확인 주의

print(train_csv.shape)  #(1459, 10)
print(test_csv.shape)   #(715, 9)
print(submission_csv.shape)     #(715, 1)

print(train_csv.columns)
# # ndex(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')
# x 는 결측치가 있어도 괜찮지만 y 는 있으면 안된다

train_csv.info() #train_csv의 정보를 알려주는 함수

################### 결측치 처리 1. 삭제 ###################
# print(train_csv.isnull().sum())
print(train_csv.isna().sum()) 

train_csv = train_csv.dropna() #dropna는 결측치의 행을 삭제해라
print(train_csv.isna().sum()) #na의 개수를 다 더해라
print(train_csv)        #[1328 rows x 10 columns]
print(train_csv.isna().sum())
print(train_csv.info()) #다시 확인 시 개수가 다 동일

print(test_csv.info())
#  test_csv 는 결측치 삭제 불가, test_csv 715 와 submission 715 가 같아야 한다.
#  그래서 결측치 삭제하지 않고, 데이터의 평균 값을 넣어준다.

test_csv = test_csv.fillna(test_csv.mean())  #fillna 채워라 #mean함수는 뭔디    #컬럼끼리만 평균을 낸다
print(test_csv.info())

x = train_csv.drop(['count'], axis=1)           # drop = 컬럼 하나를 삭제할 수 있다. #axis는 축이다 
print(x)        #[1328 rows x 9 columns]
y = train_csv['count']         # 'count' 컬럼만 넣어주세요
print(y.shape)   # (1328,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                    random_state= 12015)


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=9))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=500, batch_size=32, verbose=1, validation_split=0.3)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('로스 :', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)


# x,y,train_size=0.9, random_state= 4343  epochs=1000, batch_size=32
# 로스 : 1687.8980712890625                 1. submission_0716_1628
# r2 스코어 :  0.69598984269193

# x,y,train_size=0.8, random_state= 5183  epochs=1000, batch_size=32
# 로스 : 2336.36669921875                   2. submission_0716_1713
# r2 스코어 :  0.6916929268250431

# x,y,train_size=0.9822, random_state= 5757  epochs=1000, batch_size=32
# 로스 : 999.5776977539062
# r2 스코어 :  0.8504559017044275

#### r2 스코어 보다 로스 값이 더 중요하다!!

y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)       # (715, 1)

#################  submission.csv 만들기 // count 컬럼에 값만 넣어주면 된다 ######
submission_csv['count'] = y_submit
print(submission_csv)
print(submission_csv.shape)

submission_csv.to_csv(path + "submission_0716_2045.csv")

print('로스 :', loss)
print("r2 스코어 : ", r2)

print(hist)
print('=====================hist.history======================')
print(hist.history)
print('=====================loss=======================')
print(hist.history['loss'])
print('=====================val_loss======================')
print(hist.history['val_loss'])
print('========================================================')

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.legend(loc='upper right')
plt.title('kaggle Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()