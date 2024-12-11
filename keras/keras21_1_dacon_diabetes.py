# https://dacon.io/competitions/official/236068/mysubmission
# 풀어라

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import time

#1 데이터

path = 'C:/Users/ddong40/ai_2/_data/dacon/diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
sampleSubmission = pd.read_csv(path + 'sample_submission.csv', index_col = 0)

print(train_csv.shape) 
print(test_csv.shape)

x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, shuffle= True, random_state= 512)

#2 모델구성
model = Sequential()
model.add(Dense(128,activation='relu', input_dim=8))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3 컴파일 훈련
from sklearn.metrics import accuracy_score


model.compile(loss= 'mse', optimizer = 'adam', metrics = ['accuracy'])
start_time = time.time()
es = EarlyStopping(
    monitor= 'val_loss',
    mode = 'min',
    patience= 30,
    restore_best_weights= True    
)

model.fit(x_train, y_train, epochs= 3000, batch_size=12, verbose=1, validation_split= 0.2, callbacks=[es])
end_time = time.time()

#4 평가 예측,
loss = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)
print("로스 : ", loss[0])
print("acc : ", round(loss[1], 3))

y_pred = np.round(y_pred)

accuracy_score(y_test, y_pred)

print("acc스코어 : ", accuracy_score)
print("걸린시간 : ", round(end_time - start_time, 2), "초" )

y_submit = model.predict(test_csv)
y_submit = np.round(y_submit)

#5 파일 생성
sampleSubmission['Outcome'] = y_submit

sampleSubmission.to_csv(path+'samplesubmission_0722_1603.csv')

# 로스 :  0.1671319156885147
# acc :  0.74
# acc스코어 :  <function accuracy_score at 0x000002718AC63F70>
# 걸린시간 :  2.29 초

# 상단에 레이어 4개 추가, node개수 128개로 조정
# 로스 :  0.17390777170658112
# acc :  0.755
# acc스코어 :  <function accuracy_score at 0x000001CC17E73EE0>
# 걸린시간 :  2.33 초

# ramdom state 100 -> 512
# 로스 :  0.17070434987545013
# acc :  0.781
# acc스코어 :  <function accuracy_score at 0x000001EA850F3F70>
# 걸린시간 :  1.99 초

# batch size = 12
# 로스 :  0.16449449956417084
# acc :  0.806
# acc스코어 :  <function accuracy_score at 0x0000018B16C81F70>
# 걸린시간 :  2.84 초


# 로스 :  0.17000679671764374
# acc :  0.781
# acc스코어 :  <function accuracy_score at 0x000002223FAB4F70>
# 걸린시간 :  2.8 초