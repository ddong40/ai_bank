import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
import time

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
              [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
              [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predcit = np.array([50,60,70])

print(x.shape)

x = x.reshape(
    x.shape[0],
    x.shape[1]
    ,1
)

print(x.shape)

#2
model = Sequential()
model.add(Conv1D(10, 2, input_shape=(3, 1)))
model.add(Conv1D(10, 2))
model.add(Flatten())
# model.add(Bidirectional(LSTM(32, activation='relu'), input_shape=(3,1)))
# model.add((LSTM(32, input_shape=(3,1))))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#3 compile fit
start_time = time.time()
model.compile(loss= 'mse', optimizer='adam', metrics=['accuracy'])

# es = EarlyStopping(
#     monitor='loss',
#     mode = 'min',
#     verbose=1,
#     patience=50,
#     restore_best_weights=True
# )

import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path1 = 'C:/Users/ddong40/ai_2/_save/keras52/lstm2/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path1, 'k30_', date, '_', filename])


# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose=1,
#     save_best_only=True,
#     filepath=filepath
# )

model.fit(x, y, epochs=2000, batch_size=32, verbose=1, ) #callbacks=[es]
end_time = time.time()
model.save('C:/Users/ddong40/ai_2/_save/keras52/lstm2/save_3.h5')

#

loss = model.evaluate(x, y)
x_test = np.array([50, 60, 70]).reshape(1, 3, 1)
y_predict = model.predict(x_test)
# acc = accuracy_score(y, y_predict)

print('로스 : ', loss)
print('[50, 60, 70]의 결과 :', y_predict)
print('시간 : ', round(end_time - start_time, 3), '초')

# 로스 :  [3.491578172543086e-05, 0.0]
# [50, 60, 70]의 결과 : [[77.67054]]

# 로스 :  [0.00010294513776898384, 0.0]
# [50, 60, 70]의 결과 : [[77.00726]]

# 로스 :  [4.072708907187916e-05, 0.0]
# [50, 60, 70]의 결과 : [[79.9041]]

# 로스 :  [2.9247441489133053e-05, 0.0]
# [50, 60, 70]의 결과 : [[79.99303]]

# 로스 :  [0.0005396539927460253, 0.0]
# [50, 60, 70]의 결과 : [[79.929665]]
# 시간 :  8.719 초