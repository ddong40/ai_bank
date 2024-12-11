import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

np_path = 'C:/Users/ddong40/ai_2/_data/_save_npy/gender/'
x_train = np.load(np_path + 'keras43_01_x_train.npy' )
y_train = np.load(np_path + 'keras43_01_y_train.npy')
x_test1 = np.load(np_path + 'keras43_01_x_test.npy')
y_test1 = np.load(np_path + 'keras43_01_y_test.npy')

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=100, shuffle=True)


# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)

# 모델 구성
model = Sequential()
model.add(Conv2D(32, 2, activation='relu', input_shape = (100, 100, 3), padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(64, 2, activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(128, 2, activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(64, 2, activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))

# 컴파일 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])

start_time = time.time()

import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path1 = 'C:/Users/ddong40/ai_2/_save/keras45/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path1, 'k30_', date, '_', filename])


es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    verbose=1,
    patience=30,
    restore_best_weights=True   
)

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'min',
    verbose=1,
    save_best_only=True,
    filepath = filepath
)


model.fit(x_train, y_train, epochs=1000, batch_size=16, verbose=1, validation_split=0.25, callbacks= [es, mcp] )              

end_time = time.time()

# 평가 예측
loss = model.evaluate(x_test, y_test, batch_size=16)
y_predict = model.predict(x_test, batch_size=16)

print('로스 : ', loss[0])
print('acc : ', loss[1])
print('시간 : ', round(end_time - start_time, 2), '초')

# 로스 :  0.29976391792297363
# acc :  0.867500901222229
# 시간 :  944.158 초