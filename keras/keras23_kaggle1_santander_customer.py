#https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
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

#2 모델구성

model = Sequential()
model.add(Dense(128, activation = 'relu', input_dim = 200))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# 3 컴파일 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

start_time = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 30,
    restore_best_weights=True
)

model.fit(x_train, y_train, epochs=1000, batch_size=10, verbose=1, validation_split=0.25, callbacks=[es])

end_time = time.time()

#4 평가 예측
loss = model.evaluate(x_test, y_test)
print("로스 : ", loss[0])
print("정확도 : ", round(loss[1], 3))
print("시간", round(end_time - start_time, 2), '초')

y_pred = model.predict(x_test)
print(y_pred)

y_submit = model.predict(test_csv)
y_submit = np.round(y_submit)

sampleSubmission['target'] = y_submit

sampleSubmission.to_csv(path+'samplesubmission_0724_1520.csv') 