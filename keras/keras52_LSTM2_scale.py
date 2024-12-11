import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score


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
model.add(LSTM(32, input_shape=(3,1)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#3 compile fit

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

model.fit(x, y, epochs=1550, batch_size=16, verbose=1, ) #callbacks=[es]

model.save('C:/Users/ddong40/ai_2/_save/keras52/lstm2/save_3.h5')

#

loss = model.evaluate(x, y)
x_test = np.array([50, 60, 70]).reshape(1, 3, 1)
y_predict = model.predict(x_test)
# acc = accuracy_score(y, y_predict)

print('로스 : ', loss)
print('[50, 60, 70]의 결과 :', y_predict)