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

path = 'C:/Users/ddong40/ai/_data/bike-sharing-demand/' 

train_csv = pd.read_csv(path + "train.csv", index_col = 0)
test_csv = pd.read_csv(path + "test.csv", index_col = 0)
sampleSubmission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

x = train_csv.drop(['casual', 'registered', 'count'], axis=1 )
y = train_csv[['casual','registered']]
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, shuffle= True, random_state= 100)

#2 모델구성
model = Sequential()
model.add(Dense(10, activation= 'relu', input_dim = 8))
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
model.add(Dense(2))
#3 컴파일 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs= 500, batch_size=10)

#4 평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

# test_csv.to_csv(path+'test_csv_0718.csv')

print("loss :", loss)

#### 파일 출력 ### 


y_submit = model.predict(test_csv)
# submission_csv['casual', 'registered'] = y_predict
# casual_predict = y_submit[:,0]
# registered_predict = y_submit[:,1]
# print(casual_predict)
test_csv = test_csv.assign(casual= y_submit[:,0], registered = y_submit[:,1])
test_csv.to_csv(path + "test_columnplus.csv")
