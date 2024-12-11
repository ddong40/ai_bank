import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
path = 'C:/Users/ddong40/ai/_data/bike-sharing-demand/'


train_csv = pd.read_csv(path + "train.csv", index_col = 0)
test_csv = pd.read_csv(path + "test_columnplus.csv", index_col = 0)
sampleSubmission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

###### x와 y분리
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=10)

print(x_train.shape)

#2 모델구성
model = Sequential()
model.add(Dense(100, activation= 'relu', input_dim = 10))
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
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(1, activation= 'linear'))


#3 컴파일 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=10)

#4 평가 예측
loss = model.evaluate(x_test, y_test)
y_submit = model.predict(test_csv)
y_predict = model.predict([x_test])
r2 = r2_score(y_test, y_predict)

#5. 파일 출력
sampleSubmission['count'] = y_submit

sampleSubmission.to_csv(path+'samplesubmission_0718_1227.csv') 
print(loss)