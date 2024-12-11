import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers.merge import Concatenate, concatenate
from keras.callbacks import EarlyStopping

#1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).T
# 삼성 종가, 하이닉스 종가


# 거시기1, 거시기2, 거시기3, 거시기4

y1 = np.array(range(3001, 3101)) # 한강의 화씨 온도.
y2 = np.array(range(13001, 13101))

x3 = np.array([range(100,105), range(401,406)]).T

print(x1_datasets.shape)

x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1_datasets, y1, y2, test_size=0.2, random_state=100, shuffle=True)

#2-1 모델구성
input1 = Input(shape=(2,))
dense1 = Dense(32, activation='relu', name='bit1')(input1)
dense2 = Dense(128, activation='relu', name='bit2')(dense1)
dense3 = Dense(256, activation='relu', name='bit3')(dense2)
dense4 = Dense(128, activation='relu', name='bit4')(dense3)
dense5 = Dense(64, activation='relu', name='bit5')(dense4)
last_output = Dense(1, name='last')(dense5)
last_output2 = Dense(1, name='last2')(dense5)

model = Model(inputs=input1, outputs=[last_output, last_output2])

model.summary()

model.compile(loss='mse', optimizer='adam')


es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    verbose=1,
    patience=40,
    restore_best_weights=True
)

model.fit([x1_train], [y1_train, y2_train], epochs=2000, batch_size=8, verbose=1, validation_split=0.1, callbacks=[es])

model.save("./_save/keras62/ensemble4_전사영.h5")

loss = model.evaluate([x1_test], [y1_test, y2_test])
y_predict = model.predict([x3])

print(y_predict)
print(loss)

print(y_predict[0], y_predict[1])