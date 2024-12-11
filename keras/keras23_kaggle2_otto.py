#https://www.kaggle.com/competitions/otto-group-product-classification-challenge/data

#0.89점 이상 뽑아내기 ㅎㅎ


import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
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

print(x_test.shape)
print(y_test.shape)

#2 모델구성

model = Sequential()
model.add(Dense(512, activation= 'relu', input_dim = 93))
model.add(Dense(512, activation= 'relu'))
model.add(Dense(512, activation= 'relu'))
model.add(Dense(512, activation= 'relu'))
model.add(Dense(512, activation= 'relu'))
model.add(Dense(256, activation= 'relu'))
model.add(Dense(256, activation= 'relu'))
model.add(Dense(256, activation= 'relu'))
model.add(Dense(256, activation= 'relu'))
model.add(Dense(128, activation= 'relu'))
model.add(Dense(128, activation= 'relu'))
model.add(Dense(128, activation= 'relu'))
model.add(Dense(128, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(16, activation= 'relu'))
model.add(Dense(16, activation= 'relu'))
model.add(Dense(16, activation= 'relu'))
model.add(Dense(9, activation= 'softmax'))

#3 컴파일 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

start_time = time.time()

es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience= 30,
    restore_best_weights= True
)

model.fit(x_train, y_train, epochs = 1000, batch_size= 50, verbose = 1, validation_split=0.25, callbacks = [es])

end_time = time.time()

#4 평가 예측
loss = model.evaluate(x_test, y_test)
print('로스 값 :',loss[0])
print('정확도 : ',round(loss[1],3))
print('시간 : ', round(end_time - start_time, 2), '초')
y_pred = model.predict(x_test)
y_submit = model.predict(test_csv)
y_submit = np.round(y_submit)

sampleSubmission[['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']] = y_submit


sampleSubmission.to_csv(path + 'sampleSubmission_0724_1849.csv')

# 로스 값 : 0.6220168471336365
# 정확도 :  0.773

# 로스 값 : 0.600717306137085
# 정확도 :  0.787

# 로스 값 : 0.6098398566246033
# 정확도 :  0.792